import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from agents.tools import query_budget_data, calculate_variance, top_variances
from utils.rag_engine import load_vector_store

load_dotenv()

SYSTEM_PROMPT = """You are an expert FP&A analyst assistant. You help finance teams analyze \
budget vs. actual performance, explain variances, and draft executive commentary.

When answering:
1. Use the provided tools to retrieve actual financial data - never invent numbers.
2. For variance questions, calculate the variance, then explain the likely driver.
3. When asked for commentary, write in the style of a CFO memo: concise, factual, \
forward-looking. Use bullet points.
4. If you reference the 10-K filing, cite it (e.g., "Per the 10-K filing...").
"""

def build_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RAG tool from vector store
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_tool = create_retriever_tool(
        retriever,
        name="search_10k_filing",
        description="Search the 10-K annual filing for qualitative info on risks, MD&A, business segments, strategy."
    )

    tools = [query_budget_data, calculate_variance, top_variances, rag_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

if __name__ == "__main__":
    agent = build_agent()
    print("\n" + "="*60)
    print("🧠 FP&A Agent ready. Testing with sample query...")
    print("="*60 + "\n")
    result = agent.invoke({
        "input": "What drove the variance in Marketing in Q3-2025? Draft a 3-bullet CFO commentary."
    })
    print("\n" + "="*60)
    print("📊 AGENT RESPONSE:")
    print("="*60)
    print(result["output"])
