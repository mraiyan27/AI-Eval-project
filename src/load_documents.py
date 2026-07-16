from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("data/pdfs")

documents = loader.load()

print("=" * 50)
print(f"Total pages loaded: {len(documents)}")
print("=" * 50)

print("\nFirst page content:\n")

print(documents[0].page_content[:1000])

print("\nMetadata:")

print(documents[0].metadata)