from os import path
import streamlit as st
import pandas as pd
import json
import sys
import logging
import traceback
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.llm_helper import LLMHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
from batch.utilities.helpers.embedders.embedder_factory import EmbedderFactory
from batch.utilities.helpers.azure_search_helper import AzureSearchHelper
from batch.utilities.common.source_document import SourceDocument



sys.path.append(path.join(path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()
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



def dataframe_with_selections(df):
    if 'selected_rows' not in st.session_state:
        st.session_state['selected_rows'] = []
    selected_rows = st.multiselect("Select the document to process", options=df.index.tolist(), default=st.session_state['selected_rows'])
    st.session_state['selected_rows'] = selected_rows
    return df.loc[selected_rows]


def format_to_source_document(document_url: str, docments: List[Document]) -> List[SourceDocument]:
    source_documents: List[SourceDocument] = []
    chunk_offset = 0
    for idx, doc in enumerate(docments):
        source_documents.append(
            SourceDocument.from_metadata(
                content=doc.page_content,
                document_url=document_url,
                metadata={"offset": chunk_offset},
                idx=idx,
            )
        )
        chunk_offset += len(doc.page_content)
    return source_documents


def convert_to_search_document(document: SourceDocument):
    embedded_content = llm_helper.generate_embeddings(document.content)
    metadata = {
        "id": document.id,
        "source": document.source,
        "title": document.title,
        "chunk": document.chunk,
        "offset": document.offset,
        "page_number": document.page_number,
        "chunk_id": document.chunk_id,
    }
    return {
        "id": document.id,
        "content": document.content,
        "content_vector": embedded_content,
        "metadata": json.dumps(metadata),
        "title": document.title,
        "source": document.source,
        "chunk": document.chunk,
        "offset": document.offset,
    }


def process_document(selection_df, blob_client, embedder, azure_search_helper):
    file_name = ""
    for fd in selection_df.to_dict(orient='records'):
        file_name = fd["filename"]
        file_url = fd["fullpath"]
        file_converted = fd["converted"] if fd["converted"] else None
        file_embedings_added = fd["embeddings_added"] if fd["embeddings_added"] else None

        with st.spinner(f"Processing file: {file_name}, {file_url}"):
            if file_converted and file_name.lower().endswith('.pdf') and (file_embedings_added is None or not file_embedings_added):
                converted_file_name = f"converted/{file_name}_converted.md"
                if blob_client.file_exists(converted_file_name):
                    # Chunk the markdown file
                    with st.spinner(f"Chunking file: '{converted_file_name}' ..."):
                        md_content = blob_client.download_file(converted_file_name).decode('utf-8')
                        splits = text_splitter.split_text(md_content)
                    st.success(f"Chunked file '{converted_file_name}' into {len(splits)} chunks.")
                    # Embedding the chunks
                    with st.spinner(f"Embedding and indexing file: '{converted_file_name}' ..."):
                        source_documents = format_to_source_document(file_url, splits)
                        documents_to_upload: List[SourceDocument] = []
                        for document in source_documents:
                            documents_to_upload.append(convert_to_search_document(document))
                        response = azure_search_helper.get_search_client().upload_documents(
                            documents_to_upload
                        )
                        if not all([r.succeeded for r in response]):
                            logger.error("Failed to upload documents to search index")
                            raise Exception(response)
                        blob_client.upsert_blob_metadata(file_name, {"embeddings_added": "true"})
                    st.success(f"file '{converted_file_name}' embedd to Azure search index.")
                else:
                    embedder.embed_file(file_url, file_name)
            else:
                embedder.embed_file(file_url, file_name)
        st.success(f"file '{file_name}' has been processed.")


def main():
    st.write(
    """
    # Azure Document Intelligence
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


    try:
        # Set up Blob Storage Client
        blob_client = AzureBlobStorageClient()
        # Get all files from Blob Storage
        files_data = blob_client.get_all_files()
        # Filter out files that have already been processed
        files_data = (
            list(filter(lambda x: not x["embeddings_added"], files_data))
        )

        # Create a DataFrame from the files data
        df = pd.DataFrame(files_data)
        if df is None or df.empty:
            st.write("No documents to process.")
            st.stop()
        st.write("Documents to be processed:", df)
        selection_df = dataframe_with_selections(df)
        st.write("Your selection:")
        st.write(selection_df)

        embedder = EmbedderFactory.create(env_helper)
        azure_search_helper = AzureSearchHelper()

        need_process = not selection_df.empty
        if st.button("Start process selected documents", disabled=not need_process):
            process_document(selection_df, blob_client, embedder, azure_search_helper)
            st.session_state['selected_rows'] = []

    except Exception as e:
            logger.error(f"An error occurred while analyzing the document layout: {e}") 
            raise ValueError(f"Error: {traceback.format_exc()}. Error: {e}")


if __name__ == "__main__":
    main()