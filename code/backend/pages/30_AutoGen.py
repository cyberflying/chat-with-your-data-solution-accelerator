import os
import streamlit as st



st.set_page_config(
    page_title="Welcome",
    page_icon=os.path.join("images", "Copilot.png"),
    layout="wide",
    menu_items=None,
)

MOD_PAGE_STYLE = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(MOD_PAGE_STYLE, unsafe_allow_html=True)


col1, col2 = st.columns([1, 8])
with col1:
    st.image(os.path.join("images", "logo.png"))
with col2:
    st.write("")
    st.write("# :rainbow[Microsoft Innovation Hub]")

st.write(
    """
        # AutoGen-AgentChat: A Generalist Multi-Agent System for Solving Complex Tasks
        * **AutoGen** is an open-source framework for building AI agent systems.
        * **AgentChat** is a high-level API for building multi-agent applications.
    """
)

# st.image("https://www.microsoft.com/en-us/research/uploads/prod/2024/11/magentic_orchestrator.png", caption="AutoGen AgentChat Architecture")
st.image(os.path.join("images","AutoGen_AgentChat_Architure.jpg"), caption="AutoGen AgentChat Architecture")

st.write(
    """
        ## Teams
        * In AgentChat, teams define how groups of agents collaborate to address tasks.
        * A team is composed of one or more agents, and interacts with your application by receiving task and returning task result.
        * It is stateful and maintains context across multiple tasks.
        * A team uses a stateful termination condition to determine when to stop processing the current task.

        The diagram below shows the relationship between team and your application.
    """
)

col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment="top")
with col1:
    st.header("Round-Robin")
    st.image("https://microsoft.github.io/autogen/0.4.0.dev8/_images/agentchat-team.svg", caption="Round-Robin")
    st.page_link("pages/31_AutoGen-Travel_Planning.py", label="Demo: Travel Planning Agent", icon="👀")
    st.page_link("pages/32_AutoGen-Company_Search_and_Analysis.py", label="Demo: Company Financial Report Agent", icon="👀")
with col2:
    st.header("Selector Group Chat")
    st.image("https://microsoft.github.io/autogen/0.4.0.dev8/_images/selector-group-chat.svg", caption="Selector Group Chat")
    st.page_link("pages/33_AutoGen-Web_Search_and_Analysis.py", label="Demo: Web Search Agent", icon="👀")
with col3:
    st.header("Swarm")
    st.image("https://microsoft.github.io/autogen/0.4.0.dev8/_images/swarm_stock_research.svg", caption="Swarm")
    st.page_link("pages/34_AutoGen-Travel_Planning Pro.py", label="Demo: Travel Planning Pro Agent", icon="👀")
