import os
import json

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def evaluate_response(
    question,
    expected_answer,
    retrieved_context,
    generated_answer
):

    # Optional expected answer section (used for offline evaluation)
    expected_section = ""

    if expected_answer.strip():
        expected_section = f"""
==================================================

EXPECTED ANSWER

{expected_answer}
"""

    prompt = f"""
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Your ONLY job is to evaluate whether the generated answer is supported by the retrieved context.

NEVER use outside knowledge.

==================================================
QUESTION
==================================================

{question}

{expected_section}

==================================================
RETRIEVED CONTEXT
==================================================

{retrieved_context[:5000]}

==================================================
GENERATED ANSWER
==================================================

{generated_answer}

==================================================
EVALUATION PROCESS
==================================================

Step 1.

Break the generated answer into individual factual claims.

Step 2.

For EACH claim determine whether it is:

- Fully Supported
- Partially Supported
- Unsupported
- Contradicted

using ONLY the retrieved context.

Do NOT assume anything.

==================================================
SCORING RULES
==================================================

Correctness (1-5)

5
Every factual claim is fully supported.

4
Mostly supported.
At most ONE minor unsupported inference.

3
Multiple unsupported claims OR one important unsupported claim.

2
Major factual errors or several unsupported claims.

1
Mostly incorrect or fabricated.

IMPORTANT

If there is ANY Unsupported claim,

Correctness CANNOT be 5.

If there are MORE THAN TWO unsupported claims,

Correctness MUST NOT exceed 3.

--------------------------------------------------

Relevance (1–5)


Measures how well the generated answer addresses the user's question
while staying focused on information supported by the retrieved context.

5
Fully answers the question, stays on-topic, and contains no significant unsupported information.

4
Answers the question well but includes one minor unsupported or unnecessary point.

3
Answers only part of the question OR includes multiple unsupported/off-topic statements.

2
Only partially addresses the question and contains major unsupported or irrelevant information.

1
Fails to answer the user's question.

IMPORTANT

If hallucination = Yes AND there is at least one Major hallucination,
Relevance MUST NOT exceed 4.

If there are two or more Major hallucinations,
Relevance MUST NOT exceed 3.

If unsupported statements significantly change the answer,
Relevance MUST NOT exceed Correctness + 1.

--------------------------------------------------

Completeness (1-5)

5
All important information from the retrieved context is included.

4
One minor point missing.

3
Several important points missing.

2
Many important points missing.

1
Very incomplete.

IMPORTANT

Completeness SHOULD NOT be higher than Correctness + 1.

--------------------------------------------------

Hallucination

Return ONLY

Yes

or

No

A hallucination is ANY factual statement that is not directly supported by the retrieved documents.

==================================================
Hallucinated Parts
==================================================

Return EVERY unsupported claim.

For each provide

text

reason

severity

where severity is one of

Minor

Moderate

Major

Example

[
  {{
    "text":"GlobalLogic has 500 offices.",
    "reason":"This information does not appear anywhere in the retrieved context.",
    "severity":"Major"
  }}
]

If none

return

[]



==================================================
Explanation
==================================================

One concise sentence summarizing the evaluation.

==================================================
Return ONLY valid JSON

Example

{{
    "correctness":3,
    "relevance":5,
    "completeness":3,
    "hallucination":"Yes",

    "hallucinated_parts":[
        {{
            "text":"...",
            "reason":"...",
            "severity":"Major"
        }}
    ],

    "explanation":"..."
}}

Return JSON only.
"""

    completion = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0

    )

    output = completion.choices[0].message.content.strip()

    output = output.replace("```json", "")
    output = output.replace("```", "")
    output = output.strip()

    scores = json.loads(output)
        # -------------------------------------------------
    # Ensure optional fields always exist
    # -------------------------------------------------

    scores.setdefault("hallucinated_parts", [])
    

    hallucinations = scores["hallucinated_parts"]

    # -------------------------------------------------
    # Count hallucination severity
    # -------------------------------------------------

    major = sum(
        1
        for h in hallucinations
        if h.get("severity", "").lower() == "major"
    )

    moderate = sum(
        1
        for h in hallucinations
        if h.get("severity", "").lower() == "moderate"
    )

    minor = sum(
        1
        for h in hallucinations
        if h.get("severity", "").lower() == "minor"
    )
        # -------------------------------------------------
    # Consistency Rules
    # -------------------------------------------------

    if str(scores.get("hallucination", "")).lower() == "yes":

        # Major hallucination -> correctness cannot exceed 2
        if major >= 1:

            scores["correctness"] = min(
                scores["correctness"],
                2
            )

        # Multiple moderate hallucinations -> max 3
        elif moderate >= 2:

            scores["correctness"] = min(
                scores["correctness"],
                3
            )

        # One moderate hallucination -> max 4
        elif moderate == 1:

            scores["correctness"] = min(
                scores["correctness"],
                4
            )

        # Multiple minor hallucinations -> max 4
        elif minor >= 2:

            scores["correctness"] = min(
                scores["correctness"],
                4
            )

    # -------------------------------------------------
    # Completeness consistency
    # -------------------------------------------------

    scores["completeness"] = min(
        scores["completeness"],
        scores["correctness"] + 1
    )

    # -------------------------------------------------
    # Overall Score (weighted)
    # -------------------------------------------------

    overall_score = round(
        (
            scores["correctness"] * 0.45 +
            scores["relevance"] * 0.30 +
            scores["completeness"] * 0.25
        ),
        2
    )

    scores["overall_score"] = overall_score

    # -------------------------------------------------
    # Debug
    # -------------------------------------------------

    print("\nEvaluation Result")
    print(json.dumps(scores, indent=4))

    return scores