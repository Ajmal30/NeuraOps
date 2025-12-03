from typing import TypedDict, List 
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os 
import requests

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
# groq/compound
llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state 

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Please Enter your message: ")
while user_input != "end":
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter: ")


# llm = ChatGroq(model="groq/compound")
# response = llm.invoke("Explain LangGraph in simple words")
# print(response.content)