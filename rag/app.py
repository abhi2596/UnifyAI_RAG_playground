import streamlit as st 
from unify import Unify
from llama_index.core import VectorStoreIndex, Settings,Document
from llama_index.core.embeddings import resolve_embed_model
# from dotenv import load_dotenv
from unify_llm import Unify
from PyPDF2 import PdfReader
from llama_index.core.node_parser import SentenceSplitter
import unify

# load_dotenv()

def reset():
    st.session_state.messages = []

st.title("Chat with Data")

api_key = st.sidebar.text_input("Unify AI Key",type="password")

model_name = st.sidebar.selectbox("Select Model",options=unify.list_models(),index=1,on_change=reset)

provider_name = st.sidebar.selectbox("Select a Provider",options=unify.list_providers(model_name),on_change=reset)

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", accept_multiple_files=False,on_change=reset)

clear = st.sidebar.button("Clear Chat",on_click=reset)

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text_list = []
    for page in reader.pages:
        text_list.append(page.extract_text())
    documents = [Document(text=t) for t in text_list]

    Settings.embed_model = resolve_embed_model(embed_model="local:BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_documents(documents)

    Settings.llm = Unify(model=f"{model_name}@{provider_name}",api_key=api_key)

    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context",llm=Settings.llm,verbose=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        initial = " Follow this steps before asking questions about your data \n 1. Enter Unify API key \n 2. Select a Model and Provider using the sidebar \n 3. Upload a PDF file which is not encrypted \n 4. Any changes to sidebar will reset chat"
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
        response = chat_engine.stream_chat(prompt)
        response = st.write_stream(response.response_gen)
        # response = st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})