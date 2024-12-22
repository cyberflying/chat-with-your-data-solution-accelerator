import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient
import sys
from os import path
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.llm_helper import LLMHelper


# page layout configuration
sys.path.append(path.join(path.dirname(__file__), ".."))
st.set_page_config(
    page_title="Call Center",
    page_icon=path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load the common CSS
load_css("pages/common.css")


env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()

credential = DefaultAzureCredential()
cosmos_endpoint = f"https://{env_helper.AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/"
cosmos_database_name = env_helper.AZURE_COSMOSDB_DATABASE
cosmos_container_name = "CallTranscripts"


def make_cosmos_db_vector_search_request(query_embedding, max_results=5,minimum_similarity_score=0.5):
    """Create and return a new vector search request. Key assumptions:
    - Query embedding is a list of floats based on a search string.
    - Cosmos DB endpoint, key, and database name stored in Streamlit secrets."""

    # Create a CosmosClient
    client = CosmosClient(url=cosmos_endpoint, credential=credential)
    # Load the Cosmos database and container
    database = client.get_database_client(cosmos_database_name)
    container = database.get_container_client(cosmos_container_name)

    results = container.query_items(
        query=f"""
            SELECT TOP {max_results}
                c.id,
                c.call_id,
                c.call_transcript,
                c.abstractive_summary,
                VectorDistance(c.request_vector, @request_vector) AS SimilarityScore
            FROM c
            WHERE
                VectorDistance(c.request_vector, @request_vector) > {minimum_similarity_score}
            ORDER BY
                VectorDistance(c.request_vector, @request_vector)
            """,
        parameters=[
            {"name": "@request_vector", "value": query_embedding}
        ],
        enable_cross_partition_query=True
    )

    # Create and return a new vector search request
    return results



def main():
    """Main function for the call center search dashboard."""

    st.write(
    """
    # Call Center Transcript Search

    This Streamlit dashboard is intended to support vector search as part
    of a call center monitoring solution. It is not intended to be a
    production-ready application.
    """
    )

    st.write("## Search for Text")

    query = st.text_input("Query:", key="query")
    max_results = st.number_input("Max Results:", min_value=1, max_value=10, value=5)
    minimum_similarity_score = st.slider("Minimum Similarity Score:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if st.button("Submit"):
        with st.spinner("Searching transcripts..."):
            if query:
                query_embedding = llm_helper.generate_embeddings(query)
                response = make_cosmos_db_vector_search_request(query_embedding, max_results, minimum_similarity_score)
                for item in response:
                    st.write(item)
                st.success("Transcript search completed successfully.")
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
