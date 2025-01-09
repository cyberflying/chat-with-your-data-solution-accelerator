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


async def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


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



    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=az_model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            Web search agent: Searches for information
            Data analyst: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="A web search agent.",
        tools=[search_web_tool],
        model_client=az_model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="A data analyst agent. Useful for performing calculations.",
        model_client=az_model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        """,
    )

    def selector_func(messages: Sequence[AgentMessage]) -> str | None:
        if messages[-1].source != planning_agent.name:
            return planning_agent.name
        return None

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("TERMINATE")
    # Define a termination condition that stops the task after 5 messages.
    max_message_termination = MaxMessageTermination(20)
    # Combine the termination conditions using the `|` operator so that the
    # task stops when either condition is met.
    termination = text_termination | max_message_termination
    team = SelectorGroupChat([planning_agent, web_search_agent, data_analyst_agent], model_client=az_model_client, termination_condition=termination, selector_func=selector_func,)


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
