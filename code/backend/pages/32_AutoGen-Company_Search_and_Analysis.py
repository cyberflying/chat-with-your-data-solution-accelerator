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


async def analyze_stock(ticker: str) -> dict:  # type: ignore[type-arg]
    import os
    from datetime import datetime, timedelta

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from pytz import timezone  # type: ignore

    stock = yf.Ticker(ticker)

    # Get historical data (1 year of data to ensure we have enough for 200-day MA)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)

    # Ensure we have data
    if hist.empty:
        return {"error": "No historical data available for the specified ticker."}

    # Compute basic statistics and additional metrics
    current_price = stock.info.get("currentPrice", hist["Close"].iloc[-1])
    year_high = stock.info.get("fiftyTwoWeekHigh", hist["High"].max())
    year_low = stock.info.get("fiftyTwoWeekLow", hist["Low"].min())

    # Calculate 50-day and 200-day moving averages
    ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
    ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

    # Calculate YTD price change and percent change
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
    ytd_data = hist.loc[ytd_start:]  # type: ignore[misc]
    if not ytd_data.empty:
        price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
        percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
    else:
        price_change = percent_change = np.nan

    # Determine trend
    if pd.notna(ma_50) and pd.notna(ma_200):
        if ma_50 > ma_200:
            trend = "Upward"
        elif ma_50 < ma_200:
            trend = "Downward"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient data for trend analysis"

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Create result dictionary
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "52_week_high": year_high,
        "52_week_low": year_low,
        "50_day_ma": ma_50,
        "200_day_ma": ma_200,
        "ytd_price_change": price_change,
        "ytd_percent_change": percent_change,
        "trend": trend,
        "volatility": volatility,
    }

    # Convert numpy types to Python native types for better JSON serialization
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()

    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist["Close"], label="Close Price")
    plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA")
    plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA")
    plt.title(f"{ticker} Stock Price (Past Year)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Save plot to file
    os.makedirs("finrpt", exist_ok=True)
    plot_file_path = f"finrpt/{ticker}_stockprice.png"
    plt.savefig(plot_file_path)
    result["plot_file_path"] = plot_file_path
    st.image(result["plot_file_path"])

    return result


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

    stock_analysis_agent = AssistantAgent(
        name="Stock_Analysis_Agent",
        model_client=az_model_client,
        tools=[analyze_stock],
        description="Analyze stock data and generate a plot",
        system_message="You are a helpful AI assistant. Solve tasks using your tools. Analyze stock data and generate a plot.",
    )

    report_agent = AssistantAgent(
        name="Report_Agent",
        model_client=az_model_client,
        description="Generate a report based on the search and stock analysis results",
        system_message="""
        You are a helpful assistant that can generate a comprehensive report on a given topic based on search and stock analysis.
        Display the stock plot at "plot_file_path".
        When you done with generating the report, reply with TERMINATE.
        """,
    )


    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("TERMINATE")
    # Define a termination condition that stops the task after 5 messages.
    max_message_termination = MaxMessageTermination(20)
    # Combine the termination conditions using the `|` operator so that the
    # task stops when either condition is met.
    termination = text_termination | max_message_termination
    team = RoundRobinGroupChat([search_agent, stock_analysis_agent, report_agent], termination_condition=termination)


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
