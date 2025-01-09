import streamlit as st
import sys
from os import path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import AgentMessage
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio
from typing import Sequence
import requests
import os
# from dotenv import load_dotenv
from batch.utilities.helpers.env_helper import EnvHelper
from batch.utilities.helpers.llm_helper import LLMHelper


sys.path.append(path.join(path.dirname(__file__), ".."))
st.set_page_config(
    page_title="Travel Planning Agent",
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


# load_dotenv()
env_helper: EnvHelper = EnvHelper()
llm_helper = LLMHelper()

azure_open_model_info = os.getenv("AZURE_OPENAI_MODEL_INFO")
azure_openai_embedding_model_info = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_INFO")
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


async def clear_chat_data():
    st.session_state.messages = []


async def search_web_tool(query: str, num_results: int = 3) -> list:
    subscription_key = os.getenv("BING_API_KEY")
    bing_endpoint = os.getenv("BING_ENDPOINT")
    search_url = f"{bing_endpoint}v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params  = {"q": query, "count": num_results}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    enriched_results = []
    for result in search_results["webPages"]["value"]:
        enriched_results.append({"title": result["name"], "url": result["url"], "abstract:": result["snippet"]})
    return enriched_results


async def arxiv_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Search Arxiv for papers and return the results including abstracts.
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    for paper in client.results(search):
        results.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
            }
        )

    # # Write results to a file
    # with open('arxiv_search_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)

    return results


async def main():

    az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment = env_helper.AZURE_OPENAI_MODEL,
        model = env_helper.AZURE_OPENAI_MODEL_NAME,
        api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
        # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint = f"https://{os.getenv('AZURE_OPENAI_RESOURCE')}.openai.azure.com/",
        azure_ad_token_provider = token_provider,
        model_capabilities={
            "vision": True,
            "function_calling": True,
            "json_output": True,
        },
    )



    search_agent = AssistantAgent(
        name="Search_Agent",
        model_client=az_model_client,
        tools=[search_web_tool],
        description="Search web for information, returns top 3 results with a snippet and body content",
        system_message="You are a helpful AI assistant. Solve tasks using your tools.",
    )

    arxiv_search_agent = AssistantAgent(
        name="Arxiv_Search_Agent",
        tools=[arxiv_search],
        model_client=az_model_client,
        description="An agent that can search Arxiv for papers related to a given topic, including abstracts",
        system_message="You are a helpful AI assistant. Solve tasks using your tools. Specifically, you can take into consideration the user's request and craft a search query that is most likely to return relevant academi papers.",
    )

    report_agent = AssistantAgent(
        name="Report_Agent",
        model_client=az_model_client,
        description="Generate a report based on a given topic",
        system_message="""
        You are a helpful assistant.
        Your task is to synthesize data extracted into a high quality literature review including CORRECT references.
        You MUST write a final report that is formatted as a literature review with CORRECT references.
        Your response should end with the word 'TERMINATE'.
        """,
    )


    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("TERMINATE")
    # Define a termination condition that stops the task after 5 messages.
    max_message_termination = MaxMessageTermination(20)
    # Combine the termination conditions using the `|` operator so that the
    # task stops when either condition is met.
    termination = text_termination | max_message_termination
    team = RoundRobinGroupChat([search_agent, arxiv_search_agent, report_agent], termination_condition=termination)


    if question := st.chat_input("Please enter your travel plan"):
        async for message in team.run_stream(task=question):
            if isinstance(message, TaskResult):
                st.markdown(f"Stop Reason: {message.stop_reason}")
            else:
                if message.source == 'user':
                    with st.chat_message('user'):
                        st.markdown(message.content)
                else:
                    with st.chat_message('assistant'):
                        st.markdown(message.source)
                        st.markdown(message.content)
                        st.markdown(message.models_usage)

    clear_chat = st.button("Clear chat", key="clear_chat", on_click=clear_chat_data)



if __name__ == "__main__":
    asyncio.run(main())
