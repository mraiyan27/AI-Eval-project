import time
from dotenv import load_dotenv

from retriever import retrieve
from generator import generate_answer
from logger import save_chat
from config import USE_PHOENIX

# ----------------------------
# Optional Phoenix Tracing
# ----------------------------

if USE_PHOENIX:

    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register()

    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider
    )

load_dotenv()

print("=" * 80)
print("GlobalLogic RAG Assistant")
print("=" * 80)

from chatbot import ask_question

while True:

    question = input("\nAsk a question (type exit to quit): ")

    if question.lower() == "exit":
        break

    result = ask_question(question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print("•", source)

    print("\n" + "=" * 80)
    print("QUERY STATISTICS")
    print("=" * 80)

    print(f"Retrieved Chunks : {result['retrieved_chunks']}")
    print(f"Unique Sources   : {len(result['sources'])}")
    print(f"Latency          : {result['latency']} sec")

print("\nGoodbye!")