import streamlit as st 
import os
from unify import Unify
# from dotenv import load_dotenv

# load_dotenv()

api_key = st.sidebar.text_input("Unify AI Key",type="password")

model_name = st.sidebar.selectbox("Select Model",
                                  options=['mixtral-8x7b-instruct-v0.1',
 'llama-2-70b-chat',
 'llama-2-13b-chat',
 'mistral-7b-instruct-v0.2',
 'llama-2-7b-chat',
 'codellama-34b-instruct',
 'gemma-7b-it',
 'mistral-7b-instruct-v0.1',
 'mixtral-8x22b-instruct-v0.1',
 'codellama-13b-instruct',
 'codellama-7b-instruct',
 'yi-34b-chat',
 'llama-3-8b-chat',
 'llama-3-70b-chat',
 'pplx-7b-chat',
 'mistral-medium',
 'gpt-4',
 'pplx-70b-chat',
 'gpt-3.5-turbo',
 'deepseek-coder-33b-instruct',
 'gemma-2b-it',
 'gpt-4-turbo',
 'mistral-small',
 'mistral-large',
 'mixtral-8x7b-instruct-v0.1',
 'llama-2-70b-chat',
 'llama-2-13b-chat',
 'mistral-7b-instruct-v0.2',
 'llama-2-7b-chat',
 'codellama-34b-instruct',
 'gemma-7b-it',
 'mistral-7b-instruct-v0.1',
 'mixtral-8x22b-instruct-v0.1',
 'codellama-13b-instruct',
 'codellama-7b-instruct',
 'yi-34b-chat',
 'llama-3-8b-chat',
 'llama-3-70b-chat',
 'pplx-7b-chat',
 'mistral-medium',
 'gpt-4',
 'pplx-70b-chat',
 'gpt-3.5-turbo',
 'deepseek-coder-33b-instruct',
 'gemma-2b-it',
 'gpt-4-turbo',
 'mistral-small',
 'mistral-large'],index=44)

provider_name = st.sidebar.selectbox("Select a Provider",
                                options=["anyscale","replicate","together-ai","deepinfra","fireworks-ai","mistral-ai","octoai"],index=2)


st.title("Chat with Data")

unify = Unify(
    # This is the default and optional to include.
    api_key=api_key,
    model=model_name,
    provider=provider_name
)

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
        stream = unify.generate(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[1:]
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})