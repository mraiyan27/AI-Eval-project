import time
import pandas as pd

from retriever import retrieve
from generator import generate_answer
from judge import evaluate_response

print("=" * 60)
print("Running RAG Evaluation")
print("=" * 60)

questions_df = pd.read_csv("evaluations/questions.csv")

results = []

for index, row in questions_df.iterrows():

    question = row["question"]
    expected_answer = row["expected_answer"]
    category = row["category"]

    print("\n" + "=" * 60)
    print(f"Question {index+1}/{len(questions_df)}")
    print(question)

    start_time = time.time()

    # ----------------------------
    # Retrieve
    # ----------------------------

    docs = retrieve(question)

    retrieved_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # ----------------------------
    # Generate Answer
    # ----------------------------

    generated_answer = generate_answer(
        question,
        docs
    )

    latency = round(
        time.time() - start_time,
        2
    )

    # ----------------------------
    # Judge
    # ----------------------------

    scores = evaluate_response(

        question=question,

        expected_answer=expected_answer,

        retrieved_context=retrieved_context,

        generated_answer=generated_answer

    )

    results.append({

        "question": question,

        "category": category,

        "expected_answer": expected_answer,

        "generated_answer": generated_answer,

        "correctness": scores["correctness"],

        "relevance": scores["relevance"],

        "completeness": scores["completeness"],

        "hallucination": scores["hallucination"],

        "explanation": scores["explanation"],

        "latency_seconds": latency,

        "retrieved_chunks": len(docs)

    })

    pd.DataFrame(results).to_csv(
        "evaluations/scored_results.csv",
        index=False
    )

    print("✓ Completed")

print("\n" + "=" * 60)
print("Evaluation Finished!")
print("=" * 60)

results_df = pd.DataFrame(results)

print(results_df[[
    "question",
    "correctness",
    "relevance",
    "completeness",
    "hallucination"
]])

print("\nAverage Scores")
print("-" * 30)

print(
    "Correctness :",
    round(results_df["correctness"].mean(), 2)
)

print(
    "Relevance   :",
    round(results_df["relevance"].mean(), 2)
)

print(
    "Completeness:",
    round(results_df["completeness"].mean(), 2)
)

hallucinations = (
    results_df["hallucination"]
    .astype(str)
    .str.lower()
    .eq("yes")
    .sum()
)

print(
    "Hallucinations:",
    hallucinations,
    "/",
    len(results_df)
)