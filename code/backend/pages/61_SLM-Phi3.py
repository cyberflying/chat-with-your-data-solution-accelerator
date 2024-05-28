import streamlit as st
from langchain_community.llms import Ollama
import sys
from os import path


# page layout configuration
sys.path.append(path.join(path.dirname(__file__), ".."))
st.set_page_config(
    page_title="SLM Phi3",
    page_icon=path.join("images", "azure.ico"),
    layout="wide",
    menu_items=None,
)
mod_page_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(mod_page_style, unsafe_allow_html=True)




def clear_chat_data():
    st.session_state.messages = []



def main():
    st.write(
    """
    # SLM Phi3
    Demonstrate an example of using **SLM Phi3**. 
    ##
    """
    )


    llm = Ollama(model="phi3")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    # Await a user message and handle the chat prompt when it comes in.
    if question := st.chat_input("Enter a message:"):
        # Append the user message to the chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Echo the user's prompt to the chat window
        with st.chat_message("user"):
            st.markdown(question)
        

        # Retrieve relevant chunks based on the question
        # retrieved_docs = retriever.get_relevant_documents(question)
        result = llm.invoke(question)
        # Append the AI message to the chat history
        st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.markdown(result)

    clear_chat = st.button("Clear chat", key="clear_chat", on_click=clear_chat_data)



if __name__ == "__main__":
    main()