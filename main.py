import asyncio
import streamlit as st
from openai import AsyncOpenAI

# -----------------------------
# 配置
# -----------------------------
st.set_page_config(page_title="多模型讨论", layout="centered")

MODELS = [
    ("专家1", "gemini-3-flash-preview"),   # 需要你按实际可用模型名调整
    ("专家2",  "gpt-5.2"),
    ("专家3",  "gemini-3-pro-preview"),
]

SYSTEM_PROMPT = """你是一个哈佛大学毕业生的专职顾问，同时被要求照顾这位毕业生。你的特点是严谨，高情商（能充分考虑并应对人际关系问题），经验丰富。
你将参与“多专家讨论”。请：
1) 在回答中保持结构化（要点/步骤/结论），避免冗长。
2) 若上下文里其他专家的观点存在冲突，请指出并给出你的判断依据。
3) 尽可能给出可操作建议或示例。
"""


# -----------------------------
# session state
# -----------------------------
if "rounds" not in st.session_state:
    # [{"user": "...", "models": {"GPT4": "...", ...}}, ...]
    st.session_state.rounds = []

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# -----------------------------
# helpers
# -----------------------------
def build_history_messages_excluding_current_round():
    """
    构建“历史消息”：只包含已完成轮次（之前轮次）的用户问题 + 各模型回答。
    关键：不包含当前轮次的任何模型输出（本轮并发生成时必须隔离）。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, r in enumerate(st.session_state.rounds[:-1], start=1):
        messages.append({"role": "user", "content": f"[Round {i}] 用户问题：\n{r['user']}".strip()})
        for model_label, ans in r.get("models", {}).items():
            messages.append(
                {"role": "assistant", "content": f"[Round {i}] {model_label} 的回答：\n{ans}".strip()}
            )
    return messages


async def stream_one_model(client: AsyncOpenAI, model_label: str, model_name: str, user_text: str, history_messages,
                           placeholder: st.delta_generator.DeltaGenerator, temperature: float = 0.7):
    """
    单个模型：流式生成并实时更新 UI。
    返回：完整文本
    """
    # 本轮隔离：仅给“历史轮次” + 本轮用户问题，不给本轮其它模型回答
    messages = list(history_messages)
    current_round_idx = len(st.session_state.rounds)
    messages.append(
        {
            "role": "user",
            "content": (
                f"[Round {current_round_idx}] 用户问题：\n{user_text.strip()}\n\n"
                f"请你以 {model_label} 的身份直接作答。"
            ),
        }
    )

    acc = ""

    try:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""

            if delta:
                acc += delta
                placeholder.markdown(acc)
                # 可选：降低 UI 刷新压力
                await asyncio.sleep(0.005)

    except Exception as e:
        acc = f"（调用失败：{e}）"
        placeholder.error(acc)

    return acc


def export_markdown():
    lines = ["# 多模型讨论记录\n"]
    for i, r in enumerate(st.session_state.rounds, start=1):
        lines.append(f"## Round {i}\n")
        lines.append("### 用户\n")
        lines.append(r["user"].strip() + "\n")
        lines.append("### 专家回答\n")
        for model_label, ans in r.get("models", {}).items():
            lines.append(f"#### {model_label}\n")
            lines.append(ans.strip() + "\n")
    return "\n".join(lines).strip() + "\n"


# -----------------------------
# UI
# -----------------------------
st.title("多模型讨论")
st.caption("三个模型并发生成；同一轮中互相不可见；三者完成后再统一写入历史。")

with st.sidebar:
    st.subheader("设置")
    st.session_state.api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
    )
    # st.markdown("模型名可在代码里 `MODELS` 自行调整。")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("清空会话", use_container_width=True):
        st.session_state.rounds = []
        st.session_state.last_user_input = ""
        st.rerun()

with col_b:
    md = export_markdown()
    st.download_button(
        "导出为 Markdown",
        data=md.encode("utf-8"),
        file_name="multi_model_discussion.md",
        mime="text/markdown",
        disabled=(len(st.session_state.rounds) == 0),
        use_container_width=True,
    )

st.divider()

# 历史展示
if st.session_state.rounds:
    st.subheader("历史对话（汇总）")
    for i, r in enumerate(st.session_state.rounds, start=1):
        st.markdown(f"### Round {i} - 用户")
        st.markdown(r["user"])
        st.markdown("---")
        for model_label, ans in r.get("models", {}).items():
            st.markdown(f"### Round {i} - {model_label}")
            st.markdown(ans)
            st.markdown("---")

st.subheader("发起新一轮")
with st.form("ask_form", clear_on_submit=False):
    user_text = st.text_area(
        "输入你的问题 / 追问",
        value=st.session_state.last_user_input,
        height=120,
        placeholder="例如：请比较三种方案的优缺点，并给出推荐。",
    )
    submit = st.form_submit_button("发送给三个模型")


# -----------------------------
# 并发生成（关键逻辑）
# -----------------------------
if submit:
    if not st.session_state.api_key.strip():
        st.error("请先在左侧填写 OpenAI API Key。")
        st.stop()

    if not user_text.strip():
        st.warning("请输入问题后再发送。")
        st.stop()

    st.session_state.last_user_input = user_text.strip()

    # 先把“本轮用户问题”入栈，但 models 暂时空着（等三者完成后再一次性写入）
    st.session_state.rounds.append({"user": user_text.strip(), "models": {}})

    # 构建历史（不包含当前轮次）
    history_messages = build_history_messages_excluding_current_round()

    st.markdown("## 本轮模型输出（并发流式）")
    tabs = st.tabs([label for label, _ in MODELS])

    # 在每个 tab 创建一个 placeholder，用于流式更新
    placeholders = []
    for i, (model_label, _) in enumerate(MODELS):
        with tabs[i]:
            st.markdown(f"### {model_label}")
            placeholders.append(st.empty())

    async def run_all():
        client = AsyncOpenAI(
            api_key=st.session_state.api_key.strip(),
            base_url="https://api.chatanywhere.tech/v1"
        )

        tasks = []
        for (model_label, model_name), ph in zip(MODELS, placeholders):
            tasks.append(
                stream_one_model(
                    client=client,
                    model_label=model_label,
                    model_name=model_name,
                    user_text=user_text,
                    history_messages=history_messages,
                    placeholder=ph,
                    temperature=0.7,
                )
            )

        # 并发等待全部完成（返回顺序与 tasks 顺序一致）
        results = await asyncio.gather(*tasks)
        return results

    # Streamlit 脚本环境里运行 asyncio
    answers = asyncio.run(run_all())

    # 三者都完成后，再统一写入本轮历史（满足你的第2点）
    for (model_label, _), ans in zip(MODELS, answers):
        st.session_state.rounds[-1]["models"][model_label] = ans

    st.success("本轮三个模型已全部完成，并已统一写入历史。继续追问即可。")
    st.rerun()
