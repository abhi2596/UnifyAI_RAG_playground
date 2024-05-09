## Introduction 
RAG playground - 

Build a RAG application which will let you upload PDF's and ask questions about them. 

LLamaIndex was used to create RAG. Streamlit for interface.

Basic RAG concepts were used to develop this project:

1. Load PDF and then read text using PDFReader. 
2. Loading of data into Documents in LLamaIndex and chunking and converting them into nodes all this was done using VectorStoreIndex.from_documents
3. The text is then chunked and converted to embeddings - for embeddings BGE model was used specifically - BAAI/bge-small-en-v1.5  
4. After retreival added reranker which will reorder search results based on their relevance to a query. So after query retreival reorder happens. In this case first we will get 10 chunks based on similarity they are reranked using reranker which will give top 3.

The application was built using Streamlit and to optimize streamlit @st.experimental_fragment and @st.cache_resource were used.


## Quick Demo

![Alt Text](https://github.com/abhi2596/UnifyAI_RAG_playground/tree/main/assets/streamlit-app-2024-05-09-02-05-91.webm)

## Repository and Deployment
Github - https://github.com/abhi2596/UnifyAI_RAG_playground/tree/main
Streamlit App - https://unifyai-rag-playground.streamlit.app/

Instructions to run locally:

1. First create a virtual environment in python 

```
python -m venv <virtual env name>
```
2. Activate it and install poetry 

```
source <virtual env name>/Scripts/activate - Windows
source <virtual env name>/bin/activate - Linux/Unix
pip install poetry
```
3. Clone the repo

```
git clone https://github.com/abhi2596/UnifyAI_RAG_playground/tree/main
```
4. Run the following commands

```
poetry install 
cd rag
streamlit run app.py
```

## Contributors

| Name | GitHub Profile |
|------|----------------|
| Abhijeet Chintakunta | [abhi2596](https://github.com/abhi2596) |
