import streamlit as st
import os
from typing import TypedDict, List, Dict, Literal
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

st.set_page_config(
    page_title="Hospital Resource Management",
    page_icon="🏥",
    layout="wide"
)
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AgentState(TypedDict):
    task: str
    recommendations: str
    predictions: str
    query: str
    response: str
    error: str

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'graph' not in st.session_state:
    st.session_state.graph = None

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def get_hospital_data(hospital_name: str, raw_data: Dict[str, pd.DataFrame]) -> Dict:
    """Extract comprehensive data for a specific hospital"""
    result = {
        "hospital_info": {},
        "departments": [],
        "equipment": [],
        "staff_shifts": [],
        "alerts": [],
        "specializations": []
    }
    
    hospital_name_lower = hospital_name.lower()
    
    if 'hospitals' in raw_data:
        hospitals_df = raw_data['hospitals']
        mask = hospitals_df['hospital_name'].str.lower().str.contains(hospital_name_lower, na=False)
        if mask.any():
            result["hospital_info"] = hospitals_df[mask].iloc[0].to_dict()
            hospital_id = result["hospital_info"].get('hospital_id')
            
            if hospital_id:
                if 'departments' in raw_data:
                    dept_df = raw_data['departments']
                    result["departments"] = dept_df[dept_df['hospital_id'] == hospital_id].to_dict('records')
                
                if 'equipment_inventory' in raw_data:
                    equip_df = raw_data['equipment_inventory']
                    result["equipment"] = equip_df[equip_df['hospital_id'] == hospital_id].to_dict('records')
                
                if 'staff_shifts' in raw_data:
                    staff_df = raw_data['staff_shifts']
                    result["staff_shifts"] = staff_df[staff_df['hospital_id'] == hospital_id].to_dict('records')
                
                if 'resource_alerts' in raw_data:
                    alerts_df = raw_data['resource_alerts']
                    result["alerts"] = alerts_df[alerts_df['hospital_id'] == hospital_id].to_dict('records')
                
                if 'hospital_specializations' in raw_data:
                    spec_df = raw_data['hospital_specializations']
                    result["specializations"] = spec_df[spec_df['hospital_id'] == hospital_id].to_dict('records')
    
    return result

def find_nearby_hospitals(target_hospital_id: str, raw_data: Dict[str, pd.DataFrame], max_distance_degrees: float = 0.05) -> List[Dict]:
    """Find hospitals within proximity for resource sharing"""
    if 'hospitals' not in raw_data:
        return []
    
    hospitals_df = raw_data['hospitals']
    target = hospitals_df[hospitals_df['hospital_id'] == target_hospital_id]
    
    if target.empty:
        return []
    
    target_lat = target.iloc[0]['latitude']
    target_lon = target.iloc[0]['longitude']
    
    hospitals_df['distance'] = ((hospitals_df['latitude'] - target_lat)**2 + 
                                 (hospitals_df['longitude'] - target_lon)**2)**0.5
    
    nearby = hospitals_df[
        (hospitals_df['hospital_id'] != target_hospital_id) & 
        (hospitals_df['distance'] <= max_distance_degrees)
    ].sort_values('distance')
    
    return nearby.to_dict('records')

def ingest_data_node(state: AgentState) -> AgentState:
    """Load CSV files and create vector database"""
    csv_files = [
        "departments.csv", "equipment_inventory.csv", 
        "hospital_specializations.csv", "hospitals.csv",
        "resource_alerts.csv", "staff_shifts.csv"
    ]
    
    all_docs = []
    raw_data = {}
    
    try:
        for file in csv_files:
            try:
                df = pd.read_csv(f"dataset/{file}")
                file_key = file.replace('.csv', '')
                raw_data[file_key] = df
                
                for idx, row in df.iterrows():
                    content = f"File: {file}\n" + "\n".join([f"{col}: {val}" for col, val in row.items()])
                    all_docs.append(Document(page_content=content, metadata={"source": file, "row": idx}))
            except Exception as e:
                state["error"] = f"Could not load {file}: {e}"
        
        if not all_docs:
            state["error"] = "No data could be loaded. Please check your dataset folder."
            return state
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.raw_data = raw_data
        st.session_state.data_loaded = True
        
        state["error"] = ""
        
    except Exception as e:
        state["error"] = f"Error during data ingestion: {e}"
    
    return state

def recommendations_node(state: AgentState) -> AgentState:
    """Generate resource allocation recommendations"""
    try:
        llm = get_llm()
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke("resource allocation staff equipment occupancy alerts shortage critical high")
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a hospital operations analyst. Provide CONCISE recommendations.

DATA:
{context}

Provide 5 actionable recommendations. For each:

**[Priority: CRITICAL/HIGH/MEDIUM] - [Hospital Name]**
Issue: [One sentence with numbers]
Action: [Specific step to take]
Impact: [Expected improvement]
Timeline: [When to implement]

Keep each recommendation to 3-4 lines. Be specific with numbers."""
        
        response = llm.invoke(prompt)
        state["recommendations"] = response.content
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Error generating recommendations: {e}"
    
    return state

def predictions_node(state: AgentState) -> AgentState:
    """Make predictions based on historical data"""
    try:
        llm = get_llm()
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke("staff shifts equipment occupancy rate alerts trends patterns utilization")
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a healthcare analytics expert. Provide CONCISE predictions.

DATA:
{context}

Provide 6 predictions. For each:

**[Timeframe: 7/30/90 days] - [Hospital Name]**
Prediction: [Specific shortage/issue with numbers]
Confidence: [High/Medium/Low - X%]
Why: [One sentence explaining reason]
Action: [What to do now to prevent it]

Keep each prediction to 3-4 lines. Be specific with numbers and probabilities."""
        
        response = llm.invoke(prompt)
        state["predictions"] = response.content
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Error generating predictions: {e}"
    
    return state

def chatbot_node(state: AgentState) -> AgentState:
    """Handle complex queries with structured decision-making"""
    try:
        llm = get_llm()
        query = state["query"]
        raw_data = st.session_state.raw_data
        
        hospital_name = None
        query_lower = query.lower()
        
        if raw_data and 'hospitals' in raw_data:
            for hosp_name in raw_data['hospitals']['hospital_name'].values:
                if hosp_name.lower() in query_lower:
                    hospital_name = hosp_name
                    break
     
        hospital_data = {}
        nearby_hospitals = []
        if hospital_name and raw_data:
            hospital_data = get_hospital_data(hospital_name, raw_data)
            if hospital_data.get('hospital_info'):
                hospital_id = hospital_data['hospital_info'].get('hospital_id')
                nearby_hospitals = find_nearby_hospitals(hospital_id, raw_data)
        
        query_type = "general"
        
        info_query_patterns = [
            "how many", "how much", "what is", "what are", "list", "show me",
            "tell me about", "which", "who", "when", "where"
        ]
        is_simple_info = any(pattern in query_lower for pattern in info_query_patterns)
        
        if any(kw in query_lower for kw in ["day off", "leave", "time off", "absence", "vacation", "sick"]):
            query_type = "scheduling"
        elif any(kw in query_lower for kw in ["transfer", "move", "reallocate", "share", "borrow", "loan"]):
            query_type = "reallocation"
        elif any(kw in query_lower for kw in ["should", "approve", "deny", "recommend", "suggest", "decision"]):
            query_type = "decision"
        elif is_simple_info:
            query_type = "simple_info"
        
        context_parts = []
        
        if hospital_data:
            context_parts.append(f"=== HOSPITAL DATA FOR {hospital_name} ===")
            
            hosp_info = hospital_data.get('hospital_info', {})
            context_parts.append(f"\nHOSPITAL: {hosp_info.get('hospital_name')} ({hosp_info.get('hospital_id')})")
            context_parts.append(f"Size: {hosp_info.get('hospital_size')} | Beds: {hosp_info.get('total_bed_capacity')} | Trauma: {hosp_info.get('trauma_level')}")
            
            depts = hospital_data.get('departments', [])
            if depts:
                context_parts.append(f"\nDEPARTMENTS ({len(depts)}):")
                for dept in depts:
                    context_parts.append(f"  {dept['department_name']}: {dept['bed_count']} beds, {dept['occupancy_rate']}% occupied")
            
            equipment = hospital_data.get('equipment', [])
            if equipment:
                relevant_equip = equipment
                equipment_keywords = ['ventilator', 'wheelchair', 'stretcher', 'bed', 'mri', 'ct', 'xray', 'ultrasound']
                found_keyword = next((kw for kw in equipment_keywords if kw in query_lower), None)
                
                if found_keyword:
                    relevant_equip = [e for e in equipment if found_keyword in e['equipment_type'].lower()]
                
                relevant_equip = relevant_equip[:10]
                
                context_parts.append(f"\nEQUIPMENT (showing {len(relevant_equip)} relevant):")
                for equip in relevant_equip:
                    context_parts.append(f"  {equip['equipment_type']}: Total={equip['total_quantity']}, Available={equip['available_quantity']}, InUse={equip['in_use_quantity']}, Maintenance={equip['maintenance_quantity']}")
            
            staff_shifts = hospital_data.get('staff_shifts', [])
            if staff_shifts:
                relevant_staff = staff_shifts
                role_keywords = ['nurse', 'physician', 'doctor', 'surgeon', 'technician', 'therapist']
                found_role = next((kw for kw in role_keywords if kw in query_lower), None)
                
                if found_role:
                    relevant_staff = [s for s in staff_shifts if found_role in s['staff_role'].lower()]
                relevant_staff = relevant_staff[:15]
                
                context_parts.append(f"\nSTAFF & SHIFTS (showing {len(relevant_staff)} relevant):")
                for staff in relevant_staff:
                    context_parts.append(f"  {staff['staff_role']} - {staff['shift_name']}: {staff['staff_count']} on shift (total: {staff['total_staff_in_role']})")
            
            alerts = hospital_data.get('alerts', [])
            if alerts:
                context_parts.append(f"\nACTIVE ALERTS ({len(alerts)}):")
                for alert in alerts:
                    context_parts.append(f"  [{alert['severity']}] {alert['alert_type']}: {alert['alert_message']}")
            
            if nearby_hospitals:
                context_parts.append(f"\n=== NEARBY HOSPITALS ({len(nearby_hospitals[:3])} shown) ===")
                for hosp in nearby_hospitals[:3]:
                    context_parts.append(f"{hosp['hospital_name']} - {hosp['hospital_size']}, {hosp['total_bed_capacity']} beds, ~{hosp['distance']:.3f}° away")
        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context_parts.append("\n=== ADDITIONAL RELEVANT DATA ===")
        for doc in docs:
            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            context_parts.append(content)
        
        full_context = "\n".join(context_parts)
        
        if query_type == "scheduling":
            prompt = f"""You are a hospital operations manager making a scheduling decision.

DATA:
{full_context}

REQUEST: {query}

ANALYZE IN 5 STEPS:

1. CURRENT STATE
- Role & shift affected
- Current staff count (exact number)

2. RISK ASSESSMENT
- Patient load
- New staff-to-patient ratio
- Risk level (LOW/MEDIUM/HIGH/CRITICAL)

3. TWO OPTIONS (Name each, provide details)
**Option A:**
Coverage from: [where]
Cost: [overtime estimate]
Feasibility: [HIGH/MED/LOW]

**Option B:** [similar format]


4. DECISION: [APPROVE/DENY/CONDITIONAL]



Be specific with numbers and names."""

        elif query_type == "reallocation":
            prompt = f"""You are a resource coordinator making transfer decisions.

DATA:
{full_context}

REQUEST: {query}

ANALYSIS:

1. RESOURCE GAP
- Target hospital
- Resource & quantity needed
- Current availability

2. SOURCE OPTIONS (Top 3)
For each nearby hospital:
- Inventory & surplus
- Distance & feasibility
- Risk if transferred

3. IMPACT
**Target:** Current → After transfer (improvement)
**Source:** Remaining surplus, risk level

4. RECOMMENDATION: [Approve/Deny/Alternative]


Be specific with numbers."""

        else: 
            prompt = f"""You are a hospital information assistant. Give SHORT, DIRECT answers.

DATA:
{full_context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the specific question asked
- Use exact numbers from the data
- Be concise (2-4 sentences maximum)
- If data not available, say so clearly
- Don't provide unrequested analysis

Example good response: "City General Hospital has 30 stretchers: 18 available, 12 in use."
Example bad response: Long paragraphs about occupancy, recommendations, etc.

Answer the question directly:"""

        if st.session_state.recommendations:
            rec_summary = st.session_state.recommendations[:500]
            prompt += f"\n\nPREVIOUS RECOMMENDATIONS (summary):\n{rec_summary}"
        if st.session_state.predictions:
            pred_summary = st.session_state.predictions[:500]
            prompt += f"\n\nPREVIOUS PREDICTIONS (summary):\n{pred_summary}"
        
        response = llm.invoke(prompt)
        state["response"] = response.content
        state["error"] = ""
    except Exception as e:
        state["error"] = f"Error in chatbot: {e}"
        state["response"] = f"I encountered an error: {e}"
    
    return state

def route_task(state: AgentState) -> Literal["ingest_data", "recommendations", "predictions", "chatbot"]:
    """Route to appropriate node based on task"""
    task = state.get("task", "")
    if task == "ingest":
        return "ingest_data"
    elif task == "recommendations":
        return "recommendations"
    elif task == "predictions":
        return "predictions"
    else:
        return "chatbot"

def create_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    
    workflow.add_node("ingest_data", ingest_data_node)
    workflow.add_node("recommendations", recommendations_node)
    workflow.add_node("predictions", predictions_node)
    workflow.add_node("chatbot", chatbot_node)
    
    
    workflow.add_conditional_edges(
        START,
        route_task,
        {
            "ingest_data": "ingest_data",
            "recommendations": "recommendations",
            "predictions": "predictions",
            "chatbot": "chatbot"
        }
    )
    
    
    workflow.add_edge("ingest_data", END)
    workflow.add_edge("recommendations", END)
    workflow.add_edge("predictions", END)
    workflow.add_edge("chatbot", END)
    
    return workflow.compile()

def main():
    st.title("🏥 Hospital Resource Management System")
    st.caption("Advanced Decision Support with LangGraph & Quantitative Analysis")
    
    
    if st.session_state.graph is None:
        st.session_state.graph = create_graph()
    
    with st.sidebar:
        st.header("📊 Data Management")
        
        if st.button("🔄 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                initial_state = {
                    "task": "ingest",
                    "recommendations": "",
                    "predictions": "",
                    "query": "",
                    "response": "",
                    "error": ""
                }
                
                result = st.session_state.graph.invoke(initial_state)
                
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.success("✅ Data loaded successfully!")
        
        st.divider()
        
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            if st.session_state.raw_data:
                st.metric("Hospitals", len(st.session_state.raw_data.get('hospitals', [])))
        else:
            st.info("👆 Click to load data first")
        
        st.divider()
        
        with st.expander("💡 Query Examples"):
            st.markdown("""
**Scheduling Decisions:**
- "A surgical nurse wants a day off tomorrow at Memorial Healthcare Center. Should it be approved?"
- "ICU nurse sick at City General, who can cover the night shift?"

**Resource Allocation:**
- "Should we transfer 5 ventilators from Lakeside to Metropolitan Hospital?"
- "Which hospital can loan wheelchairs to Riverside Community?"

**Shortage Analysis:**
- "Does Bay Area Hospital have enough ICU beds for a 20% occupancy increase?"
- "Compare equipment adequacy across all Level I trauma centers"

**Strategic Questions:**
- "What happens if 3 more surgical nurses call in sick at St. Mary's?"
- "Which hospitals are most vulnerable to equipment shortages?"
            """)
    
    tab1, tab2, tab3 = st.tabs(["💡 Strategic Recommendations", "🔮 Predictive Analytics", "💬 Chat"])
    
    with tab1:
        st.header("Resource Allocation Recommendations")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first using the sidebar button.")
        else:
            if st.button("🔍 Generate Strategic Analysis", key="rec_btn", use_container_width=True):
                with st.spinner("Generating recommendations..."):
                    state = {
                        "task": "recommendations",
                        "recommendations": "",
                        "predictions": "",
                        "query": "",
                        "response": "",
                        "error": ""
                    }
                    result = st.session_state.graph.invoke(state)
                    
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.session_state.recommendations = result["recommendations"]
            
            if st.session_state.recommendations:
                st.markdown("---")
                st.markdown(st.session_state.recommendations)
    
    with tab2:
        st.header("Predictive Analytics & Forecasting")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first using the sidebar button.")
        else:
            if st.button("📊 Generate Predictive Analysis", key="pred_btn", use_container_width=True):
                with st.spinner("Analyzing patterns..."):
                    state = {
                        "task": "predictions",
                        "recommendations": "",
                        "predictions": "",
                        "query": "",
                        "response": "",
                        "error": ""
                    }
                    result = st.session_state.graph.invoke(state)
                    
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.session_state.predictions = result["predictions"]
            
            if st.session_state.predictions:
                st.markdown("---")
                st.markdown(st.session_state.predictions)
                
    with tab3:
        st.header("Chat")
        st.caption("Ask complex questions and receive structured decision analysis")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first using the sidebar button.")
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["query"])
                with st.chat_message("assistant"):
                    st.markdown(chat["response"])
            
            query = st.chat_input("Ask about scheduling, resource allocation, or operational decisions...")
            
            if query:
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data and formulating decision framework..."):
                        state = {
                            "task": "chat",
                            "recommendations": "",
                            "predictions": "",
                            "query": query,
                            "response": "",
                            "error": ""
                        }
                        result = st.session_state.graph.invoke(state)
                        
                        if result.get("error"):
                            response = f"Error: {result['error']}"
                            st.error(response)
                        else:
                            response = result["response"]
                            st.markdown(response)
                
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response
                })
                st.rerun()
            
            if st.session_state.chat_history:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col3:
                    if st.button("🗑️ Clear Chat", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()

if __name__ == "__main__":
    main()