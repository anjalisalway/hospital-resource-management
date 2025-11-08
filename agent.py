import os
import pandas as pd
import streamlit as st
from crewai import Agent, Crew, Task
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from litellm.exceptions import RateLimitError
import time

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="groq/meta-llama/llama-4-maverick-17b-128e-instruct",  # smaller model to avoid rate limits
    temperature=0.2
)

st.set_page_config(
    page_title="Hospital Resource Management",
    page_icon="🏥",
    layout="wide"
)

if "raw_data" not in st.session_state:
    st.session_state.raw_data = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {}
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

def call_llm_with_retry(prompt, max_retries=10, wait_time=5):
    """
    Call LLM with exponential backoff retry in case of rate limit errors.
    
    Parameters:
        prompt (str): The text prompt to send to the LLM.
        max_retries (int): Maximum number of retries on rate limit errors.
        wait_time (int): Base wait time in seconds, doubles on each retry.
    
    Returns:
        str: LLM-generated response.
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except RateLimitError:
            wait = wait_time * (2 ** attempt)  # exponential backoff
            st.warning(f"Rate limit hit, retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise Exception("LLM rate limit exceeded after multiple retries.")

@tool
def load_data_tool() -> dict:
    """
    Load all hospital CSV files into memory.
    
    Returns:
        dict: Each key is the CSV name (without .csv) and the value is a pandas DataFrame.
    """
    files = [
        "hospitals.csv",
        "departments.csv",
        "equipment_inventory.csv",
        "staff_shifts.csv",
        "resource_alerts.csv",
        "hospital_specializations.csv"
    ]
    raw_data = {}
    for f in files:
        try:
            raw_data[f.replace(".csv","")] = pd.read_csv(f"dataset/{f}")
        except Exception as e:
            print(f"Could not load {f}: {e}")
    return raw_data

@tool
def generate_recommendations_tool(query: str, raw_data: dict) -> str:
    """
    Generate actionable recommendations from hospital data using LLM.
    """
    prompt = f"Use hospital data to provide actionable recommendations.\n\nQUERY: {query}"
    return call_llm_with_retry(prompt)

@tool
def generate_predictions_tool(query: str, raw_data: dict) -> str:
    """
    Predict staff shortages, equipment needs, or other hospital resource requirements using LLM.
    """
    prompt = f"Use hospital data to predict shortages, staff needs, or equipment requirements.\n\nQUERY: {query}"
    return call_llm_with_retry(prompt)

@tool
def answer_query_tool(query: str, raw_data: dict) -> str:
    """
    Answer operational queries directly from CSV data to avoid rate limits.
    
    Examples:
        - "How many stretchers at City General?"
        - "Which hospital has the most ICU beds?"
    
    Returns:
        str: The answer string.
    """
    hospitals_df = raw_data.get("hospitals", pd.DataFrame())
    equipment_df = raw_data.get("equipment_inventory", pd.DataFrame())

    q_lower = query.lower()

    # Check for stretchers
    if "stretchers" in q_lower:
        hospital_name = None
        for h in hospitals_df["hospital_name"].tolist():
            if h.lower() in q_lower:
                hospital_name = h
                break
        if hospital_name:
            count = equipment_df[
                (equipment_df["hospital_name"]==hospital_name) & 
                (equipment_df["equipment"].str.lower()=="stretcher")
            ]["quantity"].sum()
            return f"{hospital_name} has {count} stretchers."
        else:
            return "Hospital not found in dataset."

    # Check for ICU beds
    if "icu beds" in q_lower:
        df = equipment_df[equipment_df["equipment"].str.lower()=="icu bed"]
        if df.empty:
            return "No ICU bed data found."
        max_row = df.loc[df["quantity"].idxmax()]
        return f"{max_row['hospital_name']} has the most ICU beds: {max_row['quantity']}."

    # Fallback to LLM for other queries
    prompt = f"Use hospital data to answer the query.\n\nQUERY: {query}"
    return call_llm_with_retry(prompt)

data_loader_agent = Agent(
    name="DataLoaderAgent",
    role="Load datasets",
    goal="Load all hospital CSVs",
    backstory="I ensure all hospital datasets are correctly imported.",
    tools=[load_data_tool],
    memory=False,
    tools_only=True,
    llm=llm
)

recommendation_agent = Agent(
    name="RecommendationAgent",
    role="Generate recommendations",
    goal="Provide actionable recommendations for hospital administration",
    backstory="I provide strategic guidance for resource optimization.",
    tools=[generate_recommendations_tool],
    memory=False,
    tools_only=True,
    llm=llm
)

prediction_agent = Agent(
    name="PredictionAgent",
    role="Generate predictions",
    goal="Predict staff shortages and equipment requirements",
    backstory="I forecast hospital needs based on trends and historical data.",
    tools=[generate_predictions_tool],
    memory=False,
    tools_only=True,
    llm=llm
)

chat_agent = Agent(
    name="ChatAgent",
    role="Answer operational queries",
    goal="Answer user queries locally using CSV data",
    backstory="I provide exact answers from hospital data without hitting the LLM unless necessary.",
    tools=[answer_query_tool],
    memory=True,
    tools_only=True,
    llm=llm
)

load_data_task = Task(
    name="LoadData",
    agent=data_loader_agent,
    tool_name="load_data_tool",
    description="Load all hospital CSVs into memory",
    expected_output="Dict of DataFrames"
)

recommendation_task = Task(
    name="GenerateRecommendations",
    agent=recommendation_agent,
    tool_name="generate_recommendations_tool",
    dependencies=["LoadData"],
    description="Generate actionable recommendations",
    expected_output="Text"
)

prediction_task = Task(
    name="GeneratePredictions",
    agent=prediction_agent,
    tool_name="generate_predictions_tool",
    dependencies=["LoadData"],
    description="Predict staff/equipment shortages",
    expected_output="Text"
)

chat_task = Task(
    name="ChatQueries",
    agent=chat_agent,
    tool_name="answer_query_tool",
    dependencies=["LoadData"],
    description="Answer operational queries",
    expected_output="Text"
)

crew = Crew(
    agents=[data_loader_agent, recommendation_agent, prediction_agent, chat_agent],
    tasks=[load_data_task, recommendation_task, prediction_task, chat_task]
)

def main():
    st.title("🏥 Hospital Resource Management System")
    
    with st.sidebar:
        st.header("📊 Data Management")
        if st.button("Load/Reload Data", use_container_width=True):
            st.session_state.raw_data = crew.kickoff(inputs={"LoadData": {}})
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully")

        st.divider()
        st.subheader("Example Queries")
        st.markdown("""
- "Generate 5 critical resource recommendations"
- "Predict staff shortages for next 30 days"
- "Does City General have enough stretchers?"
- "Which hospital has the most ICU beds?"
        """)

    
    tab1, tab2, tab3 = st.tabs(["💡 Recommendations", "🔮 Predictions", "💬 Chat"])

    with tab1:
        if not st.session_state.data_loaded:
            st.warning("Load data first")
        else:
            query = st.text_input("Enter recommendation query:", "Generate strategic recommendations")
            if st.button("Run Recommendations"):
                if query in st.session_state.recommendations:
                    st.markdown(st.session_state.recommendations[query])
                else:
                    result = crew.kickoff(inputs={"GenerateRecommendations": {"query": query, "raw_data": st.session_state.raw_data}})
                    st.session_state.recommendations[query] = result
                    st.markdown(result)

    with tab2:
        if not st.session_state.data_loaded:
            st.warning("Load data first")
        else:
            query = st.text_input("Enter prediction query:", "Predict staff/equipment shortages")
            if st.button("Run Predictions"):
                if query in st.session_state.predictions:
                    st.markdown(st.session_state.predictions[query])
                else:
                    result = crew.kickoff(inputs={"GeneratePredictions": {"query": query, "raw_data": st.session_state.raw_data}})
                    st.session_state.predictions[query] = result
                    st.markdown(result)
    with tab3:
        if not st.session_state.data_loaded:
            st.warning("Load data first")
        else:
            user_input = st.chat_input("Ask operational questions...")
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    response = crew.kickoff(inputs={"ChatQueries": {"query": user_input, "raw_data": st.session_state.raw_data}})
                    st.markdown(response)
                    st.session_state.chat_history.append({"query": user_input, "response": response})

            for msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(msg["query"])
                with st.chat_message("assistant"):
                    st.markdown(msg["response"])

            if st.session_state.chat_history and st.button("Clear Chat"):
                st.session_state.chat_history = []

if __name__ == "__main__":
    main()
