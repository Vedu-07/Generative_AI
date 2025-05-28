import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#
class DetectCallResponse(BaseModel):
    is_question_ai: bool

class CodingAIResponse(BaseModel):
    answer: str


class State(TypedDict):
    user_message: str
    ai_message: str
    is_coding_question: bool


def detect_query(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
            You are an AI assistant. Your job is to classify whether the user's question is about programming, coding, or software development.

            Respond only in valid JSON format like this:
            {"is_question_ai": true}  OR  {"is_question_ai": false}

            Examples of coding questions:
            - How do I reverse a linked list in Python?
            - What's the difference between an abstract class and interface in Java?
            - How do I deploy a Flask app to Heroku?
            - How do I create a REST API using Flask?

            Examples of non-coding/general questions:
            - What's the capital of France?
            - Tell me a joke.
            - What's the weather today?

            If the user is asking for help with programming, tools like Python, JavaScript, APIs, or technical debugging — mark it as true.
            """
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()
    response = chat.send_message(f"{SYSTEM_PROMPT}\n\nUser: {user_message}")
    
    print("Model Raw Response:", response.text)  

    try:
        parsed = json.loads(response.text)
        state["is_coding_question"] = parsed.get("is_coding_question_ai", False)
    except json.JSONDecodeError:
        state["is_coding_question"] = False
    return state


def route_edge(state: State):
    if state.get("is_coding_question"):
        return "solve_coding_question"
    else:
        return "solve_simple_question"


def solve_coding_question(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
            You are an expert coding assistant. Help the user with their programming or technical query.
            Be concise and clear.
            """
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()
    response = chat.send_message(f"{SYSTEM_PROMPT}\n\nUser: {user_message}")
    state["ai_message"] = response.text
    return state


def solve_simple_question(state: State):
    user_message = state.get("user_message")
    SYSTEM_PROMPT = """
            You are a friendly assistant. Engage in conversation and help the user with general queries.
            """
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()
    response = chat.send_message(f"{SYSTEM_PROMPT}\n\nUser: {user_message}")
    state["ai_message"] = response.text
    return state


graph_builder = StateGraph(State)
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("route_edge", route_edge)

graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)
graph_builder.add_edge("solve_coding_question", END)
graph_builder.add_edge("solve_simple_question", END)

graph = graph_builder.compile()


def call_graph():
    state = {
        "user_message": "Write code to add 2 numbers in python?",
        "ai_message": "",
        "is_coding_question": False
    }
    print("User Query:", state["user_message"])  
    result = graph.invoke(state)
    print("Final Result:", result["ai_message"])


if __name__ == "__main__":
    call_graph()
