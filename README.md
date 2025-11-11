# 📜 Papadiamantis_project
*Reconstructing the Past: How LLMs Reflect or Adapt to Papadiamantis’ Cultural Worldview*

This repository contains the full pipeline (data → chunks → embeddings → retrievers → chat UI) for exploring the literary works of Alexandros Papadiamantis using modern Retrieval-Augmented Generation (RAG). It supports genre-aware retrieval (novels, short stories, articles, poems), token-aware splitting, and an interactive Streamlit app to “talk to” Papadiamantis using different LLM backends.

It accompanies the submission to the 1st International Workshop on Language and Language Models (WoLaLa), Budapest, Hungary | November 20-21, 2025.

## 📂 Repository Structure

```
.
├── LICENSE
├── Notebooks
│   ├── 1. Scraping               # Scraping and organizing Papadiamantis' works by genre
│   │   ├── Articles.ipynb
│   │   ├── Novels.ipynb
│   │   ├── Poems.ipynb
│   │   ├── Short stories.ipynb
│   │   └── texts
│   │       ├── Άρθρα
│   │       ├── Διηγήματα
│   │       ├── Μυθιστορήματα
│   │       └── Ποιήματα
│   ├── 2.Embeddings              
│   │   ├── Embeddings.ipynb      # Preprocessing and creating embeddings and statistics for the text chunks
│   │   ├── Statistics.ipynb      # Token + chunk + embedding stats
│   │   ├── loading_documents.py
│   │   └── text_splitters.py
│   └── 3. Retrievers
│       └── Retrievers.ipynb      # Experiment with ensemble retrievers 
├── README.md
├── Streamlit App
│   ├── app.py                    # Streamlit UI (chat, sidebar settings, context viewer)
│   ├── query.py                  # RAG-aware LLM invocation (OpenAI, Gemini, Krikri model)
│   └── retriever.py              # Weaviate-backed retrievers + RankLLM reranker per genre
└── requirements.txt
```

## 🧠 What It Does

1. Loads & normalizes Papadiamantis texts
Uses loading_documents.py helper functions to read .txt, strip duplicate titles, clean footnotes, extract publication year, and attach metadata like type, theme, title, year, chapter.

2. Splits per genre (token-aware)
text_splitters.py defines different RecursiveCharacterTextSplitters for novels, short stories, articles, and poems, all using tiktoken so chunk sizes match the embedding model.

3. Embeds to a vector store (Weaviate Cluster)
Notebooks create embeddings (e.g. BAAI/bge-m3) and store them with full metadata, using Weaviate tenants per genre.

4. Builds genre-specific retrievers
retriever.py connects to Weaviate with secrets, creates 4 retrievers (novels, stories, articles, poems), and wraps them in a contextual compression retriever using Rank-LLM + gpt-4o-mini to rerank.

5. Serves everything through a Streamlit app
app.py gives you a sidebar where you choose: model (OpenAI / Gemini / Krikri), source genres, whether to use RAG context, and shows retrieved chunks under each answer.

## 🛠 Installation

The requirements.txt pins everything to the LangChain 1.0.x ecosystem and Streamlit 1.51.0:
```
langchain==1.0.5
langchain-classic==1.0.0
langchain-community==0.4.1
langchain-core==1.0.4
langchain-google-genai==3.0.1
langchain-huggingface==1.0.1
langchain-openai==1.0.2
langchain-text-splitters==1.0.0
langchain-weaviate==0.0.6
openai==2.7.1
rank-llm==0.12.8
sentence-transformers==5.1.2
streamlit==1.51.0
tiktoken==0.12.0
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
transformers==4.57.1
weaviate-client==4.17.0
```

1. Create & activate a venv (recommended)
```
python -m venv .venv
source .venv/bin/activate   # on Linux/macOS
# .venv\Scripts\activate    # on Windows
```

2. Install requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```

> Note on torch: you pinned torch==2.9.0 and torchvision==0.24.0. On some machines you may need to install the wheel that matches your CUDA / CPU setup from PyTorch’s index first, then run pip install -r requirements.txt again.

## 🔐 Secrets / Config

Create .streamlit/secrets.toml:

```
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "AIza..."
HF_TOKEN = "hf_..."
WEAVIATE_URL = "https://<your-weaviate-endpoint>"
WEAVIATE_API_KEY = "<your-key>"
```

This is used by `retriever.py` and `app.py` to connect to Weaviate and to pick the LLM backend.

## 🚀 Run the Streamlit App
```
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501/) and:

1. Pick one or more sources: Novels, Short Stories, Articles, Poems
2. Write your question (e.g. “Ποια είναι η θρησκευτικότητα στις διηγήσεις του Παπαδιαμάντη;”)
3. The app will:
    - retrieve relevant chunks from the selected tenants in Weaviate
    - compress/rerank them
    - send them to the chosen LLM
    - display the answer + retrieved context

## 📊 Notebooks: Recommended Order

- Articles.ipynb
- Short stories.ipynb
- Novels.ipynb
- Poems.ipynb
- Embeddings.ipynb
- Statistics.ipynb
- Retrievers.ipynb

## 📁 Data Layout
```
texts/
├── Μυθιστορήματα/
├── Διηγήματα/
├── Άρθρα/
└── Ποιήματα/
```

loading_documents.py walks these directories, builds Document objects, and attaches metadata like type, theme, title, and year.

## 🤝 Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, enhancements, or new features.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🪪 Citation
Mysiloglou, D. (2025). Reconstructing the Past: How LLMs Reflect or Adapt to Papadiamantis’ Cultural Worldview.
Project of the 1st International Workshop on Language and Language Models (WoLaLa), Budapest.