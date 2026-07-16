import os
import pandas as pd
from datetime import datetime

from config import CHAT_HISTORY


# -----------------------------------------
# Create CSV if it doesn't exist
# -----------------------------------------

def initialize_chat_history():

    if not os.path.exists(CHAT_HISTORY):

        columns = [

            "timestamp",

            "question",

            "retrieved_context",

            "generated_answer",

            "sources",

            "latency_seconds",

            "retrieved_chunks"

        ]

        pd.DataFrame(columns=columns).to_csv(
            CHAT_HISTORY,
            index=False
        )


# -----------------------------------------
# Save One Chat
# -----------------------------------------

def save_chat(

    question,

    docs,

    answer,

    latency

):

    initialize_chat_history()

    context = "\n\n".join(

        [doc.page_content for doc in docs]

    )

    sources = list(

        dict.fromkeys(

            [

                os.path.basename(doc.metadata["source"])

                for doc in docs

            ]

        )

    )

    new_row = {

        "timestamp":

            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "question":

            question,

        "retrieved_context":

            context,

        "generated_answer":

            answer,

        "sources":

            ", ".join(sources),

        "latency_seconds":

            round(latency, 2),

        "retrieved_chunks":

            len(docs)

    }

    df = pd.read_csv(CHAT_HISTORY)

    df.loc[len(df)] = new_row

    df.to_csv(

        CHAT_HISTORY,

        index=False

    )

# -----------------------------------------
# Save Human Evaluation
# -----------------------------------------

FEEDBACK_HISTORY = "human_feedback.csv"


def save_feedback(
    question,
    feedback,
    scores
):

    row = {

        "timestamp":
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "question":
        question,

        "feedback":
        feedback,

        "overall_score":
        scores["overall_score"],

        "correctness":
        scores["correctness"],

        "relevance":
        scores["relevance"],

        "completeness":
        scores["completeness"],

        "hallucination":
        scores["hallucination"]

    }

    if os.path.exists(FEEDBACK_HISTORY):

        df = pd.read_csv(FEEDBACK_HISTORY)

    else:

        df = pd.DataFrame(columns=row.keys())

    df.loc[len(df)] = row

    df.to_csv(
        FEEDBACK_HISTORY,
        index=False
    )