import streamlit as st
import os
import traceback
import sys
import logging
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.search.search import Search
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient
import urllib.parse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Delete Data",
    page_icon=os.path.join("images", "favicon.ico"),
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

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

try:
    search_handler = Search.get_search_handler(env_helper)

    results = search_handler.get_files()
    if results is None or results.get_count() == 0:
        st.info("No files to delete")
        st.stop()
    else:
        st.write("Select files to delete:")

    files = search_handler.output_results(results)
    selections = {
        filename: st.checkbox(filename, False, key=filename)
        for filename in files.keys()
    }
    selected_files = {
        filename: ids for filename, ids in files.items() if selections[filename]
    }

    blob_client = AzureBlobStorageClient()
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Delete embeddings only"):
            with st.spinner("Deleting files..."):
                if len(selected_files) == 0:
                    st.info("No files selected")
                    st.stop()
                else:
                    files_to_delete = search_handler.delete_files(selected_files,)
                    if len(files_to_delete) > 0:
                        filenames = files_to_delete.split(", ")
                        for f in filenames:
                            blob_name = f.split("/")[-1]
                            blob_client.upsert_blob_metadata(urllib.parse.unquote(blob_name), {"embeddings_added": "false"})
                        st.success("Deleted embeddings of files " + str(files_to_delete))
    with col2:
        if st.button("Delete embeddings and storage files"):
            with st.spinner("Deleting files..."):
                if len(selected_files) == 0:
                    st.info("No files selected")
                    st.stop()
                else:
                    files_to_delete = search_handler.delete_files(selected_files,)
                    if len(files_to_delete) > 0:
                        filenames = files_to_delete.split(", ")
                        for f in filenames:
                            blob_name = f.split("/")[-1]
                            blob_client.delete_file(urllib.parse.unquote(blob_name))
                            converted_md_blob_name = "converted/" + blob_name + "_converted.md"
                            blob_client.delete_file(urllib.parse.unquote(converted_md_blob_name))
                            i = 0
                            while True:
                                cropped_image_blob_name = "converted/" + blob_name + "_cropped_image_" + str(i) + ".png"
                                if not blob_client.file_exists(urllib.parse.unquote(cropped_image_blob_name)):
                                    break
                                blob_client.delete_file(urllib.parse.unquote(cropped_image_blob_name))
                                i += 1
                        st.success("Deleted embeddings and storage files: " + str(files_to_delete))

except Exception:
    logger.error(traceback.format_exc())
    st.error("Exception occurred deleting files.")
