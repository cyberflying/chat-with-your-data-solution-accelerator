from os import path
import streamlit as st
import sys
import logging
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, TokenTextSplitter, TextSplitter
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader, WebBaseLoader
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
from batch.utilities.helpers.azure_document_intelligence_helper import AzureDocumentIntelligenceClient
from batch.utilities.helpers.embedders.embedder_factory import EmbedderFactory

sys.path.append(path.join(path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Process Document",
    page_icon=path.join("images", "favicon.ico"),
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


# Split the document into chunks base on markdown headers.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),  
    ("#######", "Header 7"), 
    ("########", "Header 8")
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)



def main():
    st.write(
    """
    # Process Documents to the Knowledge Base
    It uses Azure Document Intelligence as document loader, which can extracts tables, paragraphs and layout information from pdf, image, office and html files. 

    The output markdown can be used in LangChain's markdown header splitter, which enables semantic chunking of the documents. 

    Then the chunked documents are indexed into vectore store, such as Azure AI Search. 
    """
    )

    # Document Intelligence Semantic Chunking in RAG architecture
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("## Document Intelligence Semantic Chunking in RAG architecture")
        st.image("https://techcommunity.microsoft.com/t5/image/serverpage/image-id/572532i131DF4C27FACF01A/image-size/large")
    with col2:
        st.write("## Advanced document processing with Azure AI Document Intelligence and Azure OpenAI services.")
        st.image("https://techcommunity.microsoft.com/t5/image/serverpage/image-id/569432iDA305FD10F7311CD/image-size/large")


    # Set up Blob Storage Client
    blob_client = AzureBlobStorageClient()
    # Get all files from Blob Storage
    files_data = blob_client.get_all_files()
    # Filter out files that have already been processed
    files_data = (
        list(filter(lambda x: not x["embeddings_added"], files_data))
    )
    st.write("not embedding files:", files_data)

    # files_data = list(map(lambda x: {"filename": x["filename"]}, files_data))
    # st.write("files name:", files_data)

    embedder = EmbedderFactory.create(env_helper)
    azure_document_intelligence_client = AzureDocumentIntelligenceClient()

    file_name = ""

    for fd in files_data:
        file_name = fd["filename"]
        file_url = fd["fullpath"]
        with st.spinner(f"Processing file: {file_name}, {file_url}"):
            # md_content = azure_document_intelligence_client.analyze_layout(file_url, file_name, gen_image_description=True)
            # converted_file_name = f"converted/{file_name}_converted.md"
            # converted_file_url = blob_client.upload_file( md_content.encode('utf-8'), converted_file_name)
            # blob_client.upsert_blob_metadata(file_name, {"converted": "true"})
            # st.success(f"Uploaded the markdown file '{converted_file_name}' to {converted_file_url}")
            
            # splits = text_splitter.split_text(md_content)
            # st.success(f"Chunked file '{converted_file_name}' into {len(splits)} chunks.")
            
            embedder.embed_file(file_url, file_name)
        st.success(f"file '{file_name}' embedd to Azure search index.")


if __name__ == "__main__":
    main()