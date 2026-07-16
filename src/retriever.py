from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    DEFAULT_K,
    BROAD_QUERY_K,
    FETCH_K,
)

# -----------------------------------------
# Embedding Model
# -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# -----------------------------------------
# Load Vector Database
# -----------------------------------------

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)


# -----------------------------------------
# Detect Broad Queries
# -----------------------------------------

def is_broad_query(question: str) -> bool:
    """
    Detect questions that require information
    from multiple documents/chunks.
    """

    broad_keywords = [
        "compare",
        "summarize",
        "summary",
        "across",
        "all documents",
        "everything",
        "overall",
        "relationship",
        "complement",
        "difference",
        "advantages and disadvantages",
        "pros and cons",
        "themes",
        "analyze",
        "analysis"
    ]

    question = question.lower()

    return any(keyword in question for keyword in broad_keywords)


# -----------------------------------------
# Retrieve Documents
# -----------------------------------------

def retrieve(question: str):

    if is_broad_query(question):
        k = BROAD_QUERY_K
    else:
        k = DEFAULT_K

    # First retrieve using MMR (diverse chunks)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": FETCH_K,
        },
    )

    docs = retriever.invoke(question)

    # Now compute relevance scores for the same query
    scored_results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    scores = [score for _, score in scored_results]

    max_score = max(scores) if scores else 0

    return docs, scores, max_score