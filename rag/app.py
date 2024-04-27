import streamlit as st 
import json
from unify import Unify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,Document
from llama_index.core.embeddings import resolve_embed_model
# from dotenv import load_dotenv
from unify_llm import Unify
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter


# load_dotenv()

st.title("Chat with Data")

api_key = st.sidebar.text_input("Unify AI Key",type="password")

model_provider = {"mixtral-8x7b-instruct-v0.1": ["together-ai", "octoai", "replicate", "mistral-ai", "perplexity-ai", "anyscale", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "llama-2-70b-chat": ["anyscale", "perplexity-ai", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "llama-2-13b-chat": ["anyscale", "together-ai", "replicate", "octoai", "fireworks-ai", "lepton-ai", "deepinfra", "aws-bedrock"], "mistral-7b-instruct-v0.2": ["perplexity-ai", "together-ai", "mistral-ai", "replicate", "aws-bedrock", "octoai", "fireworks-ai"], "llama-2-7b-chat": ["anyscale", "together-ai", "replicate", "fireworks-ai", "lepton-ai", "deepinfra"], "codellama-34b-instruct": ["anyscale", "perplexity-ai", "together-ai", "octoai", "fireworks-ai", "deepinfra"], "gemma-7b-it": ["anyscale", "together-ai", "fireworks-ai", "lepton-ai", "deepinfra"], "mistral-7b-instruct-v0.1": ["anyscale", "together-ai", "fireworks-ai", "deepinfra"], "mixtral-8x22b-instruct-v0.1": ["mistral-ai", "together-ai", "fireworks-ai", "deepinfra"], "codellama-13b-instruct": ["together-ai", "octoai", "fireworks-ai"], "codellama-7b-instruct": ["together-ai", "octoai"], "yi-34b-chat": ["together-ai", "deepinfra"], "llama-3-8b-chat": ["together-ai", "fireworks-ai"], "llama-3-70b-chat": ["together-ai", "fireworks-ai"], "pplx-7b-chat": ["perplexity-ai"], "mistral-medium": ["mistral-ai"], "gpt-4": ["openai"], "pplx-70b-chat": ["perplexity-ai"], "gpt-3.5-turbo": ["openai"], "deepseek-coder-33b-instruct": ["together-ai"], "gemma-2b-it": ["together-ai"], "gpt-4-turbo": ["openai"], "mistral-small": ["mistral-ai"], "mistral-large": ["mistral-ai"], "claude-3-haiku": ["anthropic"], "claude-3-opus": ["anthropic"], "claude-3-sonnet": ["anthropic"]}

model_name = st.sidebar.selectbox("Select Model",options=model_provider.keys(),index=20)

provider_name = st.sidebar.selectbox("Select a Provider",options=model_provider[model_name])

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", accept_multiple_files=False)

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text_list = []
    for page in reader.pages:
        text_list.append(page.extract_text())
    documents = [Document(text=t) for t in text_list]

    Settings.embed_model = resolve_embed_model(embed_model="local:BAAI/bge-small-en-v1.5")
    # build index
    index = VectorStoreIndex.from_documents(documents)

    Settings.llm = Unify(model="gemma-7b-it@anyscale",api_key=api_key)

    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        initial = "Please enter Unify API key and select a model and provider before entering a question"
        st.markdown(initial)
        st.session_state.messages.append({"role": "assistant", "content": initial})

# Accept user input
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streaming_response = chat_engine.stream_chat(prompt)
        response = st.write_stream(streaming_response.response_gen)
        st.session_state.messages.append({"role": "assistant", "content": response})