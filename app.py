import sys
import streamlit as st

# ---------------------------------------
# Allow imports from src
# ---------------------------------------

sys.path.append("src")

from chatbot import ask_question
from judge import evaluate_response
from logger import save_feedback

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="GlobalLogic RAG Evaluation Platform",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🤖 GlobalLogic RAG Evaluation Platform")

st.markdown("""
Ask questions about the GlobalLogic knowledge base and automatically evaluate
the generated answer using an independent LLM-as-a-Judge.
""")

st.divider()

# ---------------------------------------
# Sidebar
# ---------------------------------------

with st.sidebar:

    st.header("System Information")

    st.success("Generator : Gemini 2.5 Flash")
    st.success("Judge : Groq (Llama 3.3 70B)")
    st.success("Retriever : Chroma + BGE")

    st.info("""
### Pipeline

User Question

⬇️

Retriever

⬇️

Gemini

⬇️

Generated Answer

⬇️

Groq Judge

⬇️

Evaluation

⬇️

Human Feedback
""")

# ---------------------------------------
# Session State Initialization
# ---------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "question" not in st.session_state:
    st.session_state.question = ""

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ---------------------------------------
# Question Input
# ---------------------------------------

question = st.text_area(
    "Ask your question",
    height=120,
    placeholder="Example: Compare healthcare and gaming solutions."
)

# ---------------------------------------
# Generate Button
# ---------------------------------------

if st.button("Generate Answer", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # ---------------------------------------
    # Generate Answer
    # ---------------------------------------

    with st.spinner("Generating answer..."):
        result = ask_question(question)

    st.session_state.result = result
    st.session_state.question = question
    st.session_state.feedback_submitted = False

    # ---------------------------------------
    # Evaluate Answer (only if not skipped)
    # ---------------------------------------

    if not result["skip_evaluation"]:

        with st.spinner("Evaluating answer..."):
            scores = evaluate_response(
                question=question,
                expected_answer="",
                retrieved_context="\n\n".join(
                    [doc.page_content for doc in result["docs"]]
                ),
                generated_answer=result["answer"]
            )

        st.session_state.scores = scores

    else:
        st.session_state.scores = None

# ---------------------------------------
# Render Results (persisted across reruns)
# ---------------------------------------

if st.session_state.result is not None:

    result = st.session_state.result
    scores = st.session_state.scores

    # ---------------------------------------
    # Generated Answer
    # ---------------------------------------

    st.divider()
    st.subheader("💬 Generated Answer")
    st.markdown(result["answer"])

    # ---------------------------------------
    # No answer found -> skip everything else
    # ---------------------------------------

    if result["skip_evaluation"]:

        st.warning("""
The requested information is not available in the uploaded knowledge base.

Since the assistant could not answer using the retrieved documents,
automatic evaluation has been skipped.
""")

    else:

        # ---------------------------------------
        # Retrieved Evidence
        # ---------------------------------------

        st.divider()
        st.subheader("📄 Retrieved Evidence")

        for i, doc in enumerate(result["docs"], start=1):

            source = doc.metadata.get("source", "Unknown").split("\\")[-1]

            with st.expander(f"Chunk {i} • {source}"):
                st.markdown(f"**Source:** `{source}`")
                st.markdown(f"**Chunk:** {i}")
                st.text(doc.page_content)

        # ---------------------------------------
        # Retrieval Statistics
        # ---------------------------------------

        st.divider()
        st.subheader("📊 Retrieval Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Latency", f"{result['latency']} sec")

        with col2:
            st.metric("Retrieved Chunks", result["retrieved_chunks"])

        # ---------------------------------------
        # Overall Evaluation
        # ---------------------------------------

        overall_score = round(
            (
                scores["correctness"]
                + scores["relevance"]
                + scores["completeness"]
            ) / 3,
            2
        )

        st.divider()
        st.subheader("⭐ Overall Evaluation")

        st.metric("Overall Score", f"{overall_score}/5")
        st.progress(overall_score / 5)

        # ---------------------------------------
        # Detailed Metrics
        # ---------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Correctness", f"{scores['correctness']}/5")
            st.progress(scores["correctness"] / 5)

        with col2:
            st.metric("Relevance", f"{scores['relevance']}/5")
            st.progress(scores["relevance"] / 5)

        with col3:
            st.metric("Completeness", f"{scores['completeness']}/5")
            st.progress(scores["completeness"] / 5)

        with col4:
            if str(scores["hallucination"]).lower() == "no":
                st.success("✅ No Hallucination")
            else:
                st.error("❌ Hallucination Detected")

        # ---------------------------------------
        # Evaluation Summary
        # ---------------------------------------

        st.divider()
        st.markdown("### 📝 Evaluation Summary")
        st.info(scores["explanation"])

        # ---------------------------------------
        # Hallucinated Statements
        # ---------------------------------------

        hallucinated_parts = scores.get("hallucinated_parts", [])

        if hallucinated_parts:

            st.divider()
            st.subheader("🚨 Unsupported / Hallucinated Statements")

            st.warning(
                "The following statements could not be verified using the retrieved documents."
            )

            for idx, item in enumerate(hallucinated_parts, start=1):
                st.error(f"Statement {idx}")
                st.write(item.get("text", ""))
                st.caption(f"Reason: {item.get('reason', '')}")
                st.caption(f"Severity: {item.get('severity', 'Not specified')}")

        

        # ---------------------------------------
        # Metric Guide
        # ---------------------------------------

        st.divider()

        with st.expander("📖 What do these evaluation metrics mean?"):

            st.markdown("""
### ✅ Correctness (0–5)

Measures whether the generated answer is factually accurate and fully supported by the retrieved documents.

| Score | Meaning |
|-------|---------|
| **5** | Completely correct and fully supported. |
| **4** | Mostly correct with minor inaccuracies. |
| **3** | Partially correct. |
| **2** | Mostly incorrect. |
| **1** | Completely incorrect. |

---

### 🎯 Relevance (0–5)

Measures how well the generated answer addresses the user's question.

| Score | Meaning |
|-------|---------|
| **5** | Fully answers the user's question. |
| **4** | Mostly answers the question. |
| **3** | Partially answers the question. |
| **2** | Mostly irrelevant. |
| **1** | Completely irrelevant. |

---

### 📚 Completeness (0–5)

Measures whether the answer includes all important information available in the retrieved documents.

| Score | Meaning |
|-------|---------|
| **5** | Covers all important information. |
| **4** | Misses one minor point. |
| **3** | Covers only major points. |
| **2** | Misses several important points. |
| **1** | Very incomplete. |

---

### 🚫 Hallucination

A hallucination is any factual statement that **cannot be verified using the retrieved documents**.

- ✅ **No** → Every factual statement is supported by the retrieved context.
- ❌ **Yes** → One or more statements are unsupported by the retrieved context.

---

### ⭐ Overall Score

The Overall Score is calculated as:

(Correctness + Relevance + Completeness) / 3
""")

        # ---------------------------------------
        # Human Evaluation
        # ---------------------------------------

        st.divider()
        st.subheader("🙋 Human Evaluation")

        st.write("Do you agree with the automated evaluation?")

        agreement = st.radio(
            "Agreement",
            ["👍 Agree", "👎 Disagree"],
            horizontal=True,
            key="human_feedback_agreement",
            label_visibility="collapsed"
        )

        disagreement_reason = None

        if agreement == "👎 Disagree":

            st.write("What was incorrect?")

            disagreement_reason = st.radio(
                "Disagreement Reason",
                [
                    "Correctness",
                    "Relevance",
                    "Completeness",
                    "Hallucination Detection",
                    "Retrieved Evidence",
                    "Overall Evaluation",
                    "Other"
                ],
                key="human_feedback_reason",
                label_visibility="collapsed"
            )

        if st.button("Submit Feedback"):

            save_feedback(
                st.session_state.question,
                agreement,
                scores,
                
            )

            st.session_state.feedback_submitted = True

        if st.session_state.feedback_submitted:
            st.success("✅ Thank you! Your feedback has been saved.")