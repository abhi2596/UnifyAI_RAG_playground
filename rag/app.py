import streamlit as st 
from llama_index.core import VectorStoreIndex, Settings,Document
from llama_index.core.embeddings import resolve_embed_model
# from dotenv import load_dotenv
from unify_llm import Unify
from PyPDF2 import PdfReader
import unify

# load_dotenv()

# add dynamic routing

def reset():
    st.session_state.messages = []

st.title("Chat with Data")

api_key = st.sidebar.text_input("Unify AI Key",type="password")

@st.cache_data(experimental_allow_widgets=True)
def provider(model_name):
    provider_name = st.selectbox("Select a Provider",options=unify.list_providers(model_name))
    return provider_name

@st.experimental_fragment
def mp_fragment():
    model_name = st.selectbox("Select Model",options=unify.list_models(),index=7)
    provider_name = provider(model_name)
    return model_name,provider_name

@st.cache_resource 
def load_llm(model_name,provider_name):
    llm = Unify(model=f"{model_name}@{provider_name}",api_key=api_key)
    return llm

# @st.experimental_fragment()
def clear_fragment():
    st.button("Clear Chat History",on_click=reset)

with st.sidebar:
    model_name,provider_name = mp_fragment()

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", accept_multiple_files=False,on_change=reset)

with st.sidebar:
    clear_fragment()

if uploaded_file is not None:
    @st.cache_resource
    def vector_store(uploaded_file):
        reader = PdfReader(uploaded_file)
        text_list = []
        for page in reader.pages:
            text_list.append(page.extract_text())
        documents = [Document(text=t) for t in text_list]

        Settings.embed_model = resolve_embed_model(embed_model="local:BAAI/bge-small-en-v1.5")

        index = VectorStoreIndex.from_documents(documents)

        return index 
    
    index = vector_store(uploaded_file)
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context",llm=load_llm(model_name,provider_name),verbose=True,similarity_top_k=2)

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
    if api_key is None:
        if uploaded_file is None:
            st.warning("Please Enter a Unify API key and upload a file")
        else:
            st.warning("Please Enter a Unify API key")
        
    else: 
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