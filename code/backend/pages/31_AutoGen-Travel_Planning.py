import streamlit as st
import sys
from os import path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio
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

    planner_agent = AssistantAgent(
        name="planner_agent",
        model_client=az_model_client,
        description="A helpful assistant that can plan trips.",
        system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    )

    local_agent = AssistantAgent(
        name="local_agent",
        model_client=az_model_client,
        description="A local assistant that can suggest local activities or places to visit.",
        system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    )

    language_agent = AssistantAgent(
        name="language_agent",
        model_client=az_model_client,
        description="A helpful assistant that can provide language tips for a given destination.",
        system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
    )

    travel_summary_agent = AssistantAgent(
        name="travel_summary_agent",
        model_client=az_model_client,
        description="A helpful assistant that can summarize the travel plan.",
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("TERMINATE")
    # Define a termination condition that stops the task after 5 messages.
    max_message_termination = MaxMessageTermination(10)
    # Combine the termination conditions using the `|` operator so that the
    # task stops when either condition is met.
    termination = text_termination | max_message_termination
    team = RoundRobinGroupChat([planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination)


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
