from langchain_google_genai import ChatGoogleGenerativeAI

from config import GENERATION_MODEL


# -----------------------------------------
# Gemini LLM
# -----------------------------------------

llm = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL,
    temperature=0
)


# -----------------------------------------
# Build Context
# -----------------------------------------

def build_context(docs):

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context


# -----------------------------------------
# Generate Answer
# -----------------------------------------

def generate_answer(question, docs):

    context = build_context(docs)

    prompt = f"""
You are an expert GlobalLogic AI Consulting Assistant.

You must answer ONLY using the provided context.

========================
RULES
========================

1. Never use outside knowledge.

2. If the answer requires information from multiple documents,
combine the information into one coherent answer.

3. If some parts of the question are not covered by the documents,
explicitly mention which parts are missing.

4. If the answer is unavailable, reply exactly:

"I could not find that information in the provided documents."

5. For comparison questions:
- Compare both topics
- Mention similarities
- Mention differences

6. For summarization questions:
- Combine all retrieved information
- Remove duplicate points
- Organize into logical sections

7. Keep the answer well structured.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""

    response = llm.invoke(prompt)

    return response.content