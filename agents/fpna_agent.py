import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from agents.tools import (
    query_budget_data,
    calculate_variance,
    top_variances,
    summarize_all_data,
    compare_departments,
)
from utils.rag_engine import load_vector_store

load_dotenv()

SYSTEM_PROMPT = """You are an expert FP&A (Financial Planning & Analysis) analyst assistant.

YOUR DATA CONTEXT:
- You have access to budget vs. actual spending data for 5 departments (Marketing, R&D, Sales, Operations, G&A) across 8 quarters (Q1-2024 through Q4-2025).
- This is EXPENSE / SPENDING data (cost-center budgets), not revenue data.
- You also have access to a public 10-K filing for qualitative context.

CRITICAL BEHAVIOR RULES:
1. BE PROACTIVE. For vague questions like "give me an overview" or "how are we doing", IMMEDIATELY call the summarize_all_data tool and present findings. DO NOT ask the user clarifying questions when you can take a useful first action.
2. INTERPRET INTENT LIBERALLY. "revenuw", "expence", "marketting" — assume typos and proceed. Only ask for clarification if the request is fundamentally ambiguous after a tool call.
3. EDUCATE THE USER. If a user asks about "revenue" but the data is spending-focused, explain that briefly and pivot to what you CAN answer (variance analysis, overspend patterns, cost discipline).
4. ALWAYS USE TOOLS. Never invent numbers. If a tool returns no data, say so clearly and suggest what is available.
5. WRITE IN CFO MEMO STYLE. Concise, factual, bulleted, forward-looking. End with a recommendation when relevant.
6. CITE THE 10-K. When pulling qualitative context, mention "Per the 10-K filing..." so the user knows the source.

TOOL SELECTION GUIDE:
- "Overview" / "summary" / "how are we doing" -> summarize_all_data
- "Compare departments" / "which department is best" -> compare_departments
- "Variance in [dept] in [quarter]" -> calculate_variance
- "Top N variances" -> top_variances
- "Show me the data for X" -> query_budget_data
- "What does the 10-K say about Y" / "risks" / "strategy" -> search_10k_filing
"""

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_tool = create_retriever_tool(
        retriever,
        name="search_10k_filing",
        description="Search the 10-K annual filing for qualitative info on risks, MD&A, business segments, strategy, R&D, marketing.",
    )

    tools = [
        summarize_all_data,
        compare_departments,
        calculate_variance,
        top_variances,
        query_budget_data,
        rag_tool,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)

if __name__ == "__main__":
    agent = build_agent()
    print("\n🧠 Testing 'general overview' query...\n")
    result = agent.invoke({"input": "give me a general overview"})
    print("\n" + "="*60)
    print(result["output"])
