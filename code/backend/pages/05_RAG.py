import streamlit as st
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.azuresearch import AzureSearch
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.llm_helper import LLMHelper
import sys
from os import path
import logging


# page layout configuration
sys.path.append(path.join(path.dirname(__file__), ".."))
st.set_page_config(
    page_title="RAG",
    page_icon=path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("pages/common.css")


env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()
logger = logging.getLogger(__name__)

azure_openai_endpoint = env_helper.AZURE_OPENAI_ENDPOINT
azure_openai_api_key = env_helper.AZURE_OPENAI_API_KEY
azure_openai_deployment = env_helper.AZURE_OPENAI_MODEL
azure_openai_embedding_deployment = env_helper.AZURE_OPENAI_EMBEDDING_MODEL
azure_openai_api_version = env_helper.AZURE_OPENAI_API_VERSION
azure_search_endpoint: str = env_helper.AZURE_SEARCH_SERVICE
azure_search_key: str = env_helper.AZURE_SEARCH_KEY
azure_search_index_name: str = env_helper.AZURE_SEARCH_INDEX


def clear_chat_data():
    st.session_state.messages = []

def main():
    st.write(
        """
        # AOAI RAG
        Demonstrate an example of using Azure OpenAI to delvelop a RAG pattern.
        ##
        """
    )

    aoai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        azure_deployment=azure_openai_embedding_deployment,
        openai_api_version=azure_openai_api_version,
    )

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        index_name=azure_search_index_name,
        embedding_function=aoai_embeddings.embed_query,
    )

    retriever = vector_store.as_retriever(search_type="similarity")
    prompt = hub.pull("rlm/rag-prompt")
    llm = AzureChatOpenAI(
        azure_endpoint = azure_openai_endpoint,
        api_key=azure_openai_api_key,
        openai_api_version=azure_openai_api_version,
        azure_deployment=azure_openai_deployment,
        temperature=0,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Return the retrieved documents or certain source metadata from the documents
    from operator import itemgetter
    from langchain.schema.runnable import RunnableMap
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableMap(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

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
        result = rag_chain_with_source.invoke(question)
        # Append the AI message to the chat history
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})
        with st.chat_message("assistant"):
            st.markdown(result['answer'])

    clear_chat = st.button("Clear chat", key="clear_chat", on_click=clear_chat_data)


if __name__ == "__main__":
    main()
