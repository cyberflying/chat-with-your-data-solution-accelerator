import streamlit as st
import os
import traceback
import sys
import logging
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.search.search import Search
from batch.utilities.helpers.azure_blob_storage_client import AzureBlobStorageClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
env_helper: EnvHelper = EnvHelper()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Delete Data",
    page_icon=os.path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the common CSS
load_css("pages/common.css")


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
    # Initialize session state for selected files
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = {}

    search_handler = Search.get_search_handler(env_helper)
    results = search_handler.get_files()
    if results is None or results.get_count() == 0:
        st.info("No files to delete")
        st.stop()
    else:
        st.write("Select files to delete:")

    files = search_handler.output_results(results)
    with st.form("delete_form", clear_on_submit=True, border=False):
        selections = {
            filename: st.checkbox(filename, False, key=filename)
            for filename in files.keys()
        }
        selected_files = {
            filename: ids for filename, ids in files.items() if selections[filename]
        }

        if st.form_submit_button("Delete"):
            with st.spinner("Deleting files..."):
                if len(selected_files) == 0:
                    st.info("No files selected")
                    st.stop()
                else:
                    files_to_delete = search_handler.delete_files(
                        selected_files,
                    )
                    blob_client = AzureBlobStorageClient()
                    blob_client.delete_files(
                        selected_files,
                        env_helper.AZURE_SEARCH_USE_INTEGRATED_VECTORIZATION,
                    )
                    # when delete one slected file, need to delete converted .md file and figure files in converted folder.
                    # files_to_delete:  /kmdocs/rank.png, /kmdocs/北京2023年GDP-P1.pdf, /kmdocs/Hotels.txt
                    container_client = blob_client.blob_service_client.get_container_client(blob_client.container_name)
                    # list all files in the container
                    for f in container_client.list_blobs():
                        # list all files selected to delete
                        for del_f in files_to_delete.split(","):
                            # get the selected file name, not include the path, e.g. /kmdocs/, only the file name, e.g. 北京2023年GDP-P1.pdf
                            del_f_name = del_f.split("/")[-1]
                            if f.name.startswith(f"converted/{del_f_name}"):
                                blob_client.delete_file(f.name)

                    if len(files_to_delete) > 0:
                        st.success("Deleted files: " + str(files_to_delete))
                        st.rerun()
except Exception:
    logger.error(traceback.format_exc())
    st.error("Exception occurred deleting files.")
