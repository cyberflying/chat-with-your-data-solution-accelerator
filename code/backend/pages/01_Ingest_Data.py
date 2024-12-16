from os import path
import re
import streamlit as st
import traceback
import requests
import urllib.parse
import sys
import logging
from batch.utilities.helpers.config.config_helper import ConfigHelper
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
import json
from batch.utilities.helpers.llm_helper import LLMHelper
from batch.utilities.helpers.embedders.embedder_factory import EmbedderFactory
from batch.utilities.helpers.azure_document_intelligence_helper import AzureDocumentIntelligenceClient
from langchain.text_splitter import MarkdownHeaderTextSplitter
from batch.utilities.helpers.azure_search_helper import AzureSearchHelper
from typing import List
from batch.utilities.common.source_document import SourceDocument
from langchain_core.documents import Document


sys.path.append(path.join(path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Ingest Data",
    page_icon=path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the common CSS
load_css("pages/common.css")


def reprocess_all():
    backend_url = urllib.parse.urljoin(
        env_helper.BACKEND_URL, "/api/BatchStartProcessing"
    )
    params = {}
    if env_helper.FUNCTION_KEY is not None:
        params["code"] = env_helper.FUNCTION_KEY
        params["clientId"] = "clientKey"

    try:
        response = requests.post(backend_url, params=params)
        if response.status_code == 200:
            st.success(
                f"{response.text}\nPlease note this is an asynchronous process and may take a few minutes to complete."
            )
        else:
            st.error(f"Error: {response.text}")
    except Exception:
        st.error(traceback.format_exc())


def add_urls():
    urls = st.session_state["urls"].split("\n")
    add_url_embeddings_local(urls)


def add_url_embeddings_local(urls: list[str]):
    for url in urls:
        embedder.embed_file(url, ".url")


def sanitize_metadata_value(value):
    # Remove invalid characters
    # return re.sub(r"[^a-zA-Z0-9-_ .]", "?", value)
    return urllib.parse.quote(up.name)


def add_url_embeddings(urls: list[str]):
    params = {}
    if env_helper.FUNCTION_KEY is not None:
        params["code"] = env_helper.FUNCTION_KEY
        params["clientId"] = "clientKey"
    for url in urls:
        body = {"url": url}
        backend_url = urllib.parse.urljoin(
            env_helper.BACKEND_URL, "/api/AddURLEmbeddings"
        )
        r = requests.post(url=backend_url, params=params, json=body)
        if not r.ok:
            raise ValueError(f"Error {r.status_code}: {r.text}")
        else:
            st.success(f"Embeddings added successfully for {url}")


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


try:
    config = ConfigHelper.get_active_config_or_default()
    file_type = [
        processor.document_type for processor in config.document_processors
    ]
    blob_client = AzureBlobStorageClient()
    azure_document_intelligence_client = AzureDocumentIntelligenceClient()
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    azure_search_helper = AzureSearchHelper()
    embedder = EmbedderFactory.create(env_helper)


    with st.expander("Add documents: upload > embedder", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload a document to add it to the Azure Storage Account, compute embeddings and add them to the Azure AI Search index. Check your configuration for available document processors.",
            type=file_type,
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files is not None:
            for up in uploaded_files:
                # Check if the file has just been uploaded
                if up.name in st.session_state.get("uploaded_files", []):
                    st.warning(f"The file '{up.name}' has just been uploaded.")
                    continue  # Skip processing this file and continue with the next one
                if st.session_state.get("filename", "") != up.name:
                    # Upload a new file
                    # To read file as bytes:
                    bytes_data = up.getvalue()
                    title = sanitize_metadata_value(up.name)
                    st.session_state["filename"] = up.name
                    blob_url = blob_client.upload_file(bytes_data, up.name, metadata={"title": title})
                    st.session_state["file_url"] = urllib.parse.unquote(blob_url)
                    st.success(f"""File '{up.name}' uploaded to Azure Storage, stored at {blob_url}.""")

                    # Convert and embedd the file
                    with st.spinner(f"Embedding and indexing file: '{up.name}' ..."):
                        embedder.embed_file(st.session_state["file_url"], up.name)
                    st.success(f"file '{up.name}' embedd to Azure search index.")

                # Update the list of uploaded files in the session state
                if "uploaded_files" not in st.session_state:
                    st.session_state["uploaded_files"] = []
                st.session_state["uploaded_files"].append(up.name)
            if len(uploaded_files) > 0:
                st.success(
                    f"{len(uploaded_files)} documents uploaded. Embeddings added."
                )

    with st.expander("Add documents in Batch", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload a document to add it to the Azure Storage Account, compute embeddings and add them to the Azure AI Search index. Check your configuration for available document processors.",
            type=file_type,
            accept_multiple_files=True,
            key="file_uploader_batch",
        )
        if uploaded_files is not None:
            for up in uploaded_files:
                # To read file as bytes:
                bytes_data = up.getvalue()
                title = sanitize_metadata_value(up.name)
                if st.session_state.get("filename", "") != up.name:
                    # Upload a new file
                    st.session_state["filename"] = up.name
                    st.session_state["file_url"] = blob_client.upload_file(
                        bytes_data, up.name, metadata={"title": title}
                    )
            if len(uploaded_files) > 0:
                st.success(
                    f"{len(uploaded_files)} documents uploaded. Embeddings computation in progress. \nPlease note this is an asynchronous process and may take a few minutes to complete.\nYou can check for further details in the Azure Function logs."
                )

        col1, col2, col3 = st.columns([2, 1, 2])
        with col3:
            st.button(
                "Reprocess all documents in the Azure Storage account",
                on_click=reprocess_all,
            )

    with st.expander("Add URLs to the knowledge base", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_area(
                "Add a URLs and than click on 'Compute Embeddings'",
                placeholder="PLACE YOUR URLS HERE SEPARATED BY A NEW LINE",
                height=100,
                key="urls",
            )

        with col2:
            st.selectbox(
                "Embeddings models",
                [env_helper.AZURE_OPENAI_EMBEDDING_MODEL],
                disabled=True,
            )
            st.button(
                "Process and ingest web pages",
                on_click=add_urls,
                key="add_url",
            )

    with st.expander("Add documents step by step: upload > convert > chunk > embed(search client upload documents)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload a document to add it to the Azure Storage Account, compute embeddings and add them to the Azure AI Search index. Check your configuration for available document processors.",
            type=file_type,
            accept_multiple_files=True,
            key="file_uploader_step_by_step",
        )
        if uploaded_files is not None:
            for up in uploaded_files:
                # Check if the file has just been uploaded
                if up.name in st.session_state.get("uploaded_files", []):
                    st.warning(f"The file '{up.name}' has just been uploaded.")
                    continue  # Skip processing this file and continue with the next one
                if st.session_state.get("filename", "") != up.name:
                    # Upload a new file
                    # To read file as bytes:
                    bytes_data = up.getvalue()
                    title = sanitize_metadata_value(up.name)
                    st.session_state["filename"] = up.name
                    blob_url = blob_client.upload_file(bytes_data, up.name, metadata={"title": title})
                    st.session_state["file_url"] = urllib.parse.unquote(blob_url)
                    st.success(f"""File '{up.name}' uploaded to Azure Storage, stored at {blob_url}.""")

                    # Convert the file to markdown file
                    with st.spinner(f"Converting file '{up.name}' ..."):
                        md_content = azure_document_intelligence_client.analyze_layout(st.session_state["file_url"], up.name, gen_image_description=True)
                        converted_file_name = f"converted/{up.name}_converted.md"
                    st.success(f"Converted file '{up.name}' to the markdown file '{converted_file_name}', and uploaded it.")

                    # Chunk the markdown file
                    with st.spinner(f"Chunking file: '{converted_file_name}' ..."):
                        md_file_content = blob_client.download_file(converted_file_name).decode('utf-8')
                        splits = text_splitter.split_text(md_file_content)
                    st.success(f"Chunked file '{converted_file_name}' into {len(splits)} chunks.")

                    # Embedding the chunks
                    with st.spinner(f"Embedding and indexing file: '{converted_file_name}' ..."):
                        source_documents = format_to_source_document(st.session_state["file_url"], splits)
                        documents_to_upload: List[SourceDocument] = []
                        for document in source_documents:
                            documents_to_upload.append(convert_to_search_document(document))
                        response = azure_search_helper.get_search_client().upload_documents(
                            documents_to_upload
                        )
                        if not all([r.succeeded for r in response]):
                            logger.error("Failed to upload documents to search index")
                            raise Exception(response)
                        blob_client.upsert_blob_metadata(up.name, {"embeddings_added": "true"})
                    st.success(f"file '{converted_file_name}' embedd to Azure search index.")

                # Update the list of uploaded files in the session state
                if "uploaded_files" not in st.session_state:
                    st.session_state["uploaded_files"] = []
                st.session_state["uploaded_files"].append(up.name)
            if len(uploaded_files) > 0:
                st.success(
                    f"{len(uploaded_files)} documents uploaded. Embeddings added."
                )

except Exception:
    st.error(traceback.format_exc())
