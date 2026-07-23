# 🤖 GlobalLogic RAG Evaluation Platform

An end-to-end Retrieval-Augmented Generation (RAG) system and automated evaluation platform designed for querying GlobalLogic's internal knowledge base (PDF documents). The system utilizes **Google Gemini 2.5 Flash** for generation, and an independent **Llama 3.3 70B** model on Groq serving as an **LLM-as-a-Judge** to evaluate answers in real-time, accompanied by a Streamlit frontend for human validation.

---

## 🚀 Key Features

*   **Retrieval-Augmented Generation (RAG)**: Processes complex queries on corporate documents (such as Brand Guidelines, CSR Policy, Microservices Execution Playbook, Case Studies, etc.) inside `data/pdfs/`.
*   **Dynamic Query Routing**: Intelligently classifies broad vs. narrow questions to dynamically adjust retrieval parameters (`k` chunk retrieval) using MMR (Maximal Marginal Relevance) to maximize chunk diversity.
*   **LLM-as-a-Judge (Automated Evaluation)**: Assesses responses based on key criteria:
    *   **Correctness**: Factually accurate and fully supported by retrieved context.
    *   **Relevance**: Addresses the user's question directly.
    *   **Completeness**: Covers all critical information retrieved.
    *   **Hallucination Detection**: Explicitly identifies statements unsupported by retrieved context.
*   **Human-in-the-Loop Feedback**: Logs user agreements, disagreements, and detailed metrics to persistent CSV files for continuous alignment.
*   **Offline Batch Evaluation**: CLI tool to run a test set of questions through the RAG pipeline and measure performance statistics (average correctness, latency, and hallucination rates).

---

## 📊 Pipeline Architecture

```text
               +----------------------+
               |    User Question     |
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |  Retriever Classifier| <--- (Checks for comparison/summary key terms)
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |   Chroma DB Vector   | <--- (BGE Embeddings, MMR search)
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |   Gemini 2.5 Flash   | <--- (Generates Answer using prompt context)
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |    Groq Llama 3.3    | <--- (LLM-as-a-Judge evaluates response)
               +----------+-----------+
                          |
                          v
               +----------+-----------+
               |   Human Validation   | <--- (Streamlit Frontend captures UI feedback)
               +----------------------+
```

---

## 🛠️ Tech Stack

*   **Web Interface**: Streamlit
*   **Embeddings Model**: LangChain HuggingFace (`BAAI/bge-small-en-v1.5`)
*   **Vector Database**: Chroma DB
*   **Generation Model**: Google Gemini 2.5 Flash (`gemini-2.5-flash`) via `ChatGoogleGenerativeAI`
*   **Evaluation Model (Judge)**: Groq API (`llama-3.3-70b-versatile`)
*   **Data & Processing**: Python, Pandas, LangChain

---

## 📦 Directory Structure

```text
├── app.py                      # Streamlit frontend dashboard application
├── requirements.txt            # Project dependencies (Python packages)
├── human_feedback.csv          # Saved user review feedback data
├── .env                        # API keys and secret credentials (ignored)
│
├── data/
│   └── pdfs/                   # GlobalLogic PDF knowledge base documents
│
├── evaluations/
│   ├── chat_history.csv        # Saved conversations log
│   ├── questions.csv           # Evaluation test questions suite
│   ├── scored_results.csv      # Offline evaluation output results
│
├── src/
│   ├── chatbot.py              # Main Q&A chain driver logic
│   ├── retriever.py            # Chroma document retriever (MMR search)
│   ├── generator.py            # Gemini generation wrapper & prompts
│   ├── judge.py                # Llama 3.3 Groq judge evaluation rules
│   ├── evaluate.py             # Offline batch evaluation pipeline script
│   ├── create_vectorstore.py   # Script to ingest/chunk PDFs and build ChromaDB
│   ├── config.py               # Configuration constants & model names
│   └── logger.py               # Chat history and feedback logging functions
```

---

## ⚙️ Configuration & Installation

### 1. Prerequisites
Make sure Python 3.10+ is installed on your local environment.

### 2. Configure Credentials
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Initialize Vector Store (Ingest Documents)
Ensure your source PDFs are placed in the `data/pdfs/` directory, then run:
```bash
python src/create_vectorstore.py
```
This chunks the documents using recursive character splitting (size: 800, overlap: 150) and generates the Chroma DB indexes under `./chroma_db/`.

---

## 🖥️ Usage

### Run Streamlit UI Dashboard
Launch the interactive web portal to ask questions and see real-time evaluations:
```bash
streamlit run app.py
```

### Run Batch Evaluation Suite
To execute offline tests and save evaluation results to `evaluations/scored_results.csv`:
```bash
python src/evaluate.py
```

---

## 📐 Evaluation Metrics

The automated Groq Judge measures the performance of the generated responses against the retrieved context:

| Metric | Score Range | Description |
| :--- | :--- | :--- |
| **Correctness** | 1 - 5 | Measures whether the generated answer is factually accurate and fully supported by the retrieved documents. |
| **Relevance** | 1 - 5 | Measures how well the generated answer addresses the user's question without adding off-topic content. |
| **Completeness** | 1 - 5 | Measures whether the answer includes all important information available in the retrieved documents. |
| **Hallucination** | Yes / No | A hallucination is any factual statement that cannot be verified using the retrieved documents. |

*   **Overall Score**: Weighted average calculation `(Correctness * 0.45 + Relevance * 0.30 + Completeness * 0.25)`.