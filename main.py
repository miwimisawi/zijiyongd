import os
import time
import streamlit as st
from openai import OpenAI

# -----------------------------
# 配置
# -----------------------------
st.set_page_config(page_title="多模型讨论", layout="centered")

MODELS = [
    ("专家1", "gpt-5.2"),   # 需要你按实际可用模型名调整
    ("专家2",   "gemini-3-pro-preview"),
    ("专家3",  "gemini-3-flash-preview"),
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
def build_context_messages():
    """
    将所有历史：用户问题 + 所有模型回答，拼入上下文。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, r in enumerate(st.session_state.rounds, start=1):
        messages.append({"role": "user", "content": f"[Round {i}] 用户问题：\n{r['user']}".strip()})
        for model_label, ans in r.get("models", {}).items():
            messages.append(
                {"role": "assistant", "content": f"[Round {i}] {model_label} 的回答：\n{ans}".strip()}
            )
    return messages


def stream_chat_completion(client: OpenAI, model: str, messages: list[dict], temperature: float = 0.7):
    """
    OpenAI Chat Completions 流式输出（逐 chunk 返回）。
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    full_text = ""
    for chunk in stream:
        delta = ""
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            full_text += delta
            yield delta, full_text

    if full_text == "":
        yield "", ""


def export_markdown():
    lines = ["# 多模型讨论记录\n"]
    for i, r in enumerate(st.session_state.rounds, start=1):
        lines.append(f"## Round {i}\n")
        lines.append("### 用户\n")
        lines.append(r["user"].strip() + "\n")
        lines.append("### 专家评论\n")
        for model_label, ans in r.get("models", {}).items():
            lines.append(f"#### {model_label}\n")
            lines.append(ans.strip() + "\n")
    return "\n".join(lines).strip() + "\n"


# -----------------------------
# UI
# -----------------------------
st.title("多模型讨论")
st.caption("每轮提问后，三个模型分别生成回答；下一轮会携带所有历史模型输出作为上下文。")

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

# 历史展示：用原生 st.markdown（不做气泡）
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

if submit:
    if not st.session_state.api_key.strip():
        st.error("请先在左侧填写 OpenAI API Key。")
        st.stop()

    if not user_text.strip():
        st.warning("请输入问题后再发送。")
        st.stop()

    st.session_state.last_user_input = user_text.strip()

    # 初始化 client（每次提交时用当前 key）
    client = OpenAI(api_key=st.session_state.api_key.strip(), base_url="https://api.chatanywhere.tech/v1")

    # 新增本轮
    new_round = {"user": user_text.strip(), "models": {}}
    st.session_state.rounds.append(new_round)

    # 生成前的基础上下文（包含历史所有轮次；本轮目前只有用户问题）
    base_messages = build_context_messages()

    st.markdown("## 本轮模型输出（流式）")
    tabs = st.tabs([label for label, _ in MODELS])

    # 逐模型生成，并展示到对应 tab 内
    for idx, (model_label, model_name) in enumerate(MODELS):
        with tabs[idx]:
            st.markdown(f"### {model_label}")
            placeholder = st.empty()
            acc = ""

            # 让每个模型都“读到之前所有模型的输出”（包括本轮已生成的）
            messages = list(base_messages)

            # 注入：本轮已完成的模型输出（按生成顺序累积）
            already = st.session_state.rounds[-1]["models"]
            for prev_label, prev_ans in already.items():
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Round {len(st.session_state.rounds)}] {prev_label} 的回答：\n{prev_ans}".strip(),
                    }
                )

            # 明确指令：以该模型身份回答；可引用/反驳之前模型
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"现在请你以 {model_label} 的身份回答本轮用户问题。"
                        f"你可以引用或反驳其他模型已给出的观点（若有）。\n\n"
                        f"用户问题：\n{user_text.strip()}"
                    ),
                }
            )

            try:
                for _, full in stream_chat_completion(client, model_name, messages, temperature=0.7):
                    acc = full
                    placeholder.markdown(acc)
                    time.sleep(0.01)
            except Exception as e:
                acc = f"（调用失败：{e}）"
                placeholder.error(acc)

            # 保存本轮该模型回答
            st.session_state.rounds[-1]["models"][model_label] = acc

    st.success("本轮三个模型已完成。继续输入追问即可（会携带所有历史输出）。")
    st.rerun()