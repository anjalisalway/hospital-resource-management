# 🏥 Hospital Resource Management System

A **Streamlit-based application** to manage hospital resources, generate recommendations, and perform predictive analytics for decision support.


## Features

- **Data Management**: Load and index hospital data from CSV files.  
- **Strategic Recommendations**: Generate actionable resource allocation suggestions.  
- **Predictive Analytics**: Forecast staffing, equipment, and occupancy trends.  
- **Chat/Decision Support**: Ask questions about scheduling, resource allocation, or shortages and receive structured, data-driven answers.  


## Approach & Tech Stack

### Approach
1. **Data Ingestion**: CSV files for hospitals, staff, equipment, departments, alerts, and specializations are loaded and structured into a vector database.  
2. **Vector Embeddings**: All textual data is embedded using HuggingFace embeddings to support semantic search.  
3. **Recommendation Engine**: Uses LLM (Groq Llama-3) to generate strategic recommendations based on current resource status.  
4. **Predictive Analytics**: LLM analyzes historical and current data to forecast shortages, occupancy trends, and staffing needs.  
5. **Chat-Based Decision Support**: Users can query the system, and the agent returns structured, stepwise responses with numbers and actionable insights.  

### Tech Stack
- **Large Language Model**: Groq LLaMA 3.3  
- **Vector Database**: FAISS  
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`  
- **Data Processing**: Pandas, NumPy  
- **Environment Management**: python-dotenv  
- **Python Version**: 3.11+  

## Setup

1. **Clone the repository**:  
```bash
git clone <repo-url>
cd <repo-folder>
```

2. **Install dependencies**:  
```bash
pip install -r requirements.txt
```

3. **Prepare dataset / Add your own dataset**:  
```bash
python data_creation.py
```

4. **Add environment variables**:  
Create a .env file with your API keys (Groq, etc.)

## Usage(2 frameworks)

1. **Langgraph framework**:
```bash
streamlit run app.py
```
2. **Crewai framework**:
```bash
streamlit run agent.py
```

---
Use the sidebar to load/reload data.
Explore tabs for recommendations, predictive analytics, and chat-based decision support.
Ask questions in the chat to get structured analysis.



