import streamlit as st
import pandas as pd
import plotly.express as px
from agents.fpna_agent import build_agent

st.set_page_config(page_title="FP&A Copilot", page_icon="📊", layout="wide")

st.title("📊 FP&A Copilot")
st.caption("AI-powered Financial Planning & Analysis assistant — built with LangChain, GPT-4o-mini, and RAG over a 10-K filing")

# Sidebar — variance overview
with st.sidebar:
    st.header("📈 Variance Snapshot")
    df = pd.read_csv("data/fpna_data.csv")
    selected_q = st.selectbox("Select Quarter", df['Quarter'].unique()[::-1])
    q_data = df[df['Quarter'] == selected_q]
    fig = px.bar(
        q_data, x='Department', y='Variance_Pct',
        color='Variance_Pct', color_continuous_scale='RdYlGn_r',
        title=f"Variance % — {selected_q}",
        labels={'Variance_Pct': 'Variance (%)'}
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown(
        "An AI agent that automates variance analysis "
        "and writes CFO-style commentary in seconds. "
        "Built with LangChain + OpenAI."
    )

# Initialize agent (cached)
@st.cache_resource
def get_agent():
    return build_agent()

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example prompts
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
examples = [
    "What drove the variance in Marketing in Q3-2025?",
    "Top 3 variances in Q4-2025 with commentary",
    "What are Apple's biggest risks from the 10-K?"
]
for col, ex in zip([col1, col2, col3], examples):
    if col.button(ex, use_container_width=True):
        st.session_state.pending = ex

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
user_input = st.chat_input("Ask about budgets, variances, or the 10-K...")
if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            agent = get_agent()
            result = agent.invoke({"input": user_input})
            response = result["output"]
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
