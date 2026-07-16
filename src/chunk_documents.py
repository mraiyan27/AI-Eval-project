from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDFs
loader = PyPDFDirectoryLoader("data/pdfs")
documents = loader.load()

print(f"Original pages: {len(documents)}")

# Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

print("\n" + "="*60)
print("FIRST CHUNK")
print("="*60)

print(chunks[0].page_content)

print("\nMetadata")

print(chunks[0].metadata)