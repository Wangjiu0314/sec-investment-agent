from __future__ import annotations

import ast
import os
import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_deepseek import ChatDeepSeek

from .filing_tool import analyze_sec_filing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_TOOL_STEPS = 5

SYSTEM_PROMPT = """
你是一个用于学习 LangChain agent 的中文助手。
你可以使用工具完成计算、获取时间、保存笔记和 SEC 10-K 财报分析。
如果用户的问题适合用工具，就调用工具；如果不需要工具，就直接回答。
用户要求分析某家公司财报时，调用 analyze_sec_filing，并使用 status_only=false；工具会自动完成标题识别与区间划分、Map、Reduce 和审查，不要重复调用。
用户只要求查看已有分析状态时，调用 analyze_sec_filing，并使用 status_only=true。
财报工具返回 complete=false 时，明确说明仍需继续运行；不得假装报告已经完成。
财报工具返回 complete=true 时，必须告诉用户 artifacts.memo 中的完整报告路径。
财报工具是同步运行的；返回后没有后台任务。禁止说“仍在运行”或“稍后会自动完成”。
失败原因只能依据 diagnostics，不得猜测网络、文档格式或不存在的状态字段。
不要使用 emoji。
回答要简洁，并说明关键结果。
""".strip()


@tool
def get_current_time() -> str:
    """获取当前本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """安全计算一个基础数学表达式，例如 '(12 * 8) + 5'。"""
    try:
        return str(_safe_eval(expression))
    except Exception as exc:
        return f"计算失败: {exc}"


@tool
def save_note(text: str) -> str:
    """把一段简短笔记保存到项目根目录的 notes.txt。"""
    note = text.strip()
    if not note:
        return "没有收到要保存的笔记内容。"

    notes_path = PROJECT_ROOT / "notes.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with notes_path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {note}\n")
    return f"已保存到 {notes_path.name}"


TOOLS: list[BaseTool] = [
    get_current_time,
    calculate,
    save_note,
    analyze_sec_filing,
]
TOOLS_BY_NAME: dict[str, BaseTool] = {item.name: item for item in TOOLS}


def _safe_eval(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    binary_ops: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in binary_ops:
            raise ValueError("不支持这个运算符")
        return binary_ops[op_type](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in unary_ops:
            raise ValueError("不支持这个一元运算符")
        return unary_ops[op_type](_eval_node(node.operand))

    raise ValueError("只支持数字和基础数学运算")


def build_llm() -> ChatDeepSeek:
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def run_tool_calling_agent(user_input: str, messages: list[BaseMessage]) -> str:
    llm = build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages.append(HumanMessage(content=user_input))

    for _ in range(MAX_TOOL_STEPS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return _message_text(ai_message)

        for tool_call in tool_calls:
            tool_result = _run_one_tool_call(tool_call)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call.get("id", ""),
                )
            )

    return "工具调用次数过多，已停止。"


def _run_one_tool_call(tool_call: dict[str, Any]) -> str:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    selected_tool = TOOLS_BY_NAME.get(tool_name)

    if selected_tool is None:
        return f"工具不存在: {tool_name}"

    try:
        return str(selected_tool.invoke(tool_args))
    except Exception as exc:
        return f"工具 {tool_name} 执行失败: {exc}"


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def print_missing_key_help() -> None:
    print("没有配置 DEEPSEEK_API_KEY，无法调用 DeepSeek 模型。")
    print("请打开 E:\\wangbingjie\\agent\\.env，把 DEEPSEEK_API_KEY 改成你的真实 Key。")


def print_help() -> None:
    print("可用本地命令:")
    print("  help   显示帮助")
    print("  tools  显示可用工具")
    print("  reset  清空当前对话上下文")
    print("  exit   退出")
    print()
    print("可以直接问:")
    print("  现在几点？")
    print("  帮我计算 12 * 8 + 5")
    print("  把“今天学习了工具调用”保存成笔记")
    print("  查看 AAPL 财报分析状态")
    print("  分析 MSFT 的 10-K")


def print_tools() -> None:
    print("当前 agent 可用工具:")
    for item in TOOLS:
        print(f"  - {item.name}: {item.description}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key_here":
        print_missing_key_help()
        return

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("LangChain + DeepSeek 工具调用 agent 已启动。输入 help 查看命令，输入 exit 退出。")
    while True:
        user_input = input("\n你 > ").strip()
        if not user_input:
            continue

        command = user_input.lower()
        if command in {"exit", "quit", "q"}:
            print("Agent > 已退出。")
            break
        if command == "help":
            print_help()
            continue
        if command == "tools":
            print_tools()
            continue
        if command == "reset":
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            print("Agent > 已清空当前对话上下文。")
            continue

        answer = run_tool_calling_agent(user_input, messages)
        print(f"Agent > {answer}")
