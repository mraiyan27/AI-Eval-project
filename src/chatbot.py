import os
import time

from retriever import retrieve
from generator import generate_answer
from logger import save_chat


def ask_question(question: str):

    start_time = time.time()

    # -----------------------------------------
    # Retrieve Documents
    # -----------------------------------------

    docs, scores, max_score = retrieve(question)

    # -----------------------------------------
    # Generate Answer
    # -----------------------------------------

    answer = generate_answer(question, docs)

    latency = round(time.time() - start_time, 2)

    # -----------------------------------------
    # Decide whether evaluation should be skipped
    # -----------------------------------------

    no_answer_phrases = [
        "I could not find that information in the provided documents.",
        "I could not find relevant information in the provided documents."
    ]

    skip_evaluation = any(
        phrase.lower() in answer.lower()
        for phrase in no_answer_phrases
    )

    # -----------------------------------------
    # Save chat only if useful answer generated
    # -----------------------------------------

    if not skip_evaluation:

        save_chat(
            question=question,
            docs=docs,
            answer=answer,
            latency=latency
        )

    # -----------------------------------------
    # Sources
    # -----------------------------------------

    unique_sources = []

    if not skip_evaluation:

        for doc in docs:

            source = os.path.basename(doc.metadata["source"])

            if source not in unique_sources:
                unique_sources.append(source)

    # -----------------------------------------
    # Return
    # -----------------------------------------

    return {

        "answer": answer,

        "sources": unique_sources,

        "latency": latency,

        "retrieved_chunks": len(docs) if not skip_evaluation else 0,

        "docs": docs if not skip_evaluation else [],

        "scores": scores,

        "max_score": max_score,

        "skip_evaluation": skip_evaluation

    }