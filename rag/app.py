import streamlit as st 
from llama_index.core import VectorStoreIndex, Settings,Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.postprocessor import SentenceTransformerRerank
from unify_llm import Unify
from PyPDF2 import PdfReader
import unify
from llama_index.core.node_parser import SentenceSplitter

# load_dotenv()

# add dynamic routing
# look into caching again and also mp_fragment and load_llm and clear chat history 
# app should be fast

def reset():
    st.session_state.messages = []

st.title("Chat with Data")

api_key = st.sidebar.text_input("Unify AI Key",type="password")

def provider(model_name):
    dynamic = st.toggle("Dynamic Routing")
    if dynamic:
        providers = ["lowest-input-cost","lowest-output-cost","lowest-itl","lowest-ttft","highest-tks-per-sec"]
        provider_name = st.selectbox("Select a Provider",options=providers,index=1)
    else:
        provider_name = st.selectbox("Select a Provider",options=unify.list_providers(model_name))
    return provider_name

@st.experimental_fragment
def mp_fragment():
    model_name = st.selectbox("Select Model",options=unify.list_models(),index=7)
    provider_name = provider(model_name)
    return model_name,provider_name

def load_models(model_name,provider_name):
    Settings.llm = Unify(model=f"{model_name}@{provider_name}",api_key=api_key)

@st.cache_resource
def rerank_model():
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
    )
    return rerank

with st.sidebar:
    model_name,provider_name = mp_fragment()
    uploaded_file = st.file_uploader("Upload a PDF file", accept_multiple_files=False,on_change=reset)
    st.sidebar.button("Clear Chat History",on_click=reset)


load_models(model_name,provider_name)

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
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context",verbose=True,similarity_top_k=10,node_postprocessors=[rerank_model()])

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
    if api_key == "" and uploaded_file is None:
        st.warning("Please Enter a Unify API key and upload a file")
    elif api_key is None:
        st.warning("Please Enter a Unify API key")
    elif uploaded_file is None:
        st.warning("Upload a File")   
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