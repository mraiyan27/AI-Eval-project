from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

query = input("Ask a question: ")

results = vectorstore.similarity_search_with_score(
    query,
    k=5
)

print("\n" + "=" * 80)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nResult {i}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Source: {doc.metadata['source']}")
    print("-" * 80)
    print(doc.page_content[:600])