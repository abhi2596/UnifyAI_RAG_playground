import streamlit as st 
import os
from unify import Unify
# from dotenv import load_dotenv

# load_dotenv()

api_key = st.sidebar.text_input("Unify AI Key")

st.title("Chat with Data")

unify = Unify(
    # This is the default and optional to include.
    api_key=api_key,
    endpoint="llama-2-13b-chat@anyscale"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What do you want to know about your data"):
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
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})