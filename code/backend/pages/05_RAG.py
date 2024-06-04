import streamlit as st
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from azure.storage.blob import BlobServiceClient, generate_blob_sas
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.vectorstores.redis import Redis
from langchain.chains import RetrievalQA
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
import os
import sys
from os import path
import logging


# page layout configuration
sys.path.append(path.join(path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG",
    page_icon=path.join("images", "Copilot.png"),
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



blob_client = AzureBlobStorageClient()

azure_openai_endpoint = env_helper.AZURE_OPENAI_ENDPOINT
azure_openai_api_key = env_helper.AZURE_OPENAI_API_KEY
azure_openai_deployment = env_helper.AZURE_OPENAI_MODEL
azure_openai_embedding_deployment = env_helper.AZURE_OPENAI_EMBEDDING_MODEL
azure_openai_api_version = env_helper.AZURE_OPENAI_API_VERSION
doc_intelligence_endpoint = env_helper.AZURE_FORM_RECOGNIZER_ENDPOINT
doc_intelligence_key = env_helper.AZURE_FORM_RECOGNIZER_KEY
# azure_blob_connection_string = blob_client.connect_str
azure_blob_container_name = env_helper.AZURE_BLOB_CONTAINER_NAME
vector_store_type = "AzureSearch"
redis_url = "REDIS_URL"
redis_index_name = "REDIS_INDEX_NAME"
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

    if vector_store_type == "AzureSearch":
        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=azure_search_index_name,
            embedding_function=aoai_embeddings.embed_query,
        )
    elif vector_store_type == "Redis":
        vector_store: Redis = Redis(
            redis_url=redis_url,
            index_name=redis_index_name,
            embedding=aoai_embeddings,
        )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Use a prompt for RAG that is checked into the LangChain prompt hub (https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=989ad331-949f-4bac-9694-660074a208a7)
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
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

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
        # retrieved_docs = retriever.get_relevant_documents(question)
        result = rag_chain_with_source.invoke(question)
        # Append the AI message to the chat history
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})
        with st.chat_message("assistant"):
            st.markdown(result['answer'])

    clear_chat = st.button("Clear chat", key="clear_chat", on_click=clear_chat_data)


    
if __name__ == "__main__":
    main()