import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import asyncio
# import nest_asyncio2
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import subprocess
# import google.protobuf
from src.llm_call import MLFlowAssistant
import mlflow
from mlflow.tracking import MlflowClient


# nest_asyncio2.apply()         # ← ADD THIS (must be before any async calls)


if not os.path.exists("data"):
    st.write("="*100)
    st.write("="*100)
    st.write("running gdown")
    subprocess.run(
        "gdown --folder https://drive.google.com/drive/folders/1TUx3VzG3kypxmlHF2A2gHfevtkM8066p?usp=drive_link",
        shell=True,
        check=True
    )
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="BAAI/bge-small-en-v1.5",
        local_dir="./models/bge-small-en-v1.5"
    )
    st.write("gdown complete")
    st.write("="*100)
    st.write("="*100)

    # subprocess.run(
    # "curl -LsSf https://astral.sh/uv/install.sh | sh",
#     shell=True,
#     check=True
# # )

#     with open("INITIALIZED.txt", "w") as f:
#         f.write("INITIALIZED")
    
# def run(coro):              
#     return asyncio.get_event_loop().run_until_complete(coro)
#     return asyncio.run(coro)

# def run(coro):
#     try:
#         loop = asyncio.get_running_loop()
#         return loop.run_until_complete(coro)
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop.run_until_complete(coro)

# ─────────────────────────────
# MCP servers: local math via uv + fastmcp
# ─────────────────────────────
# SERVERS = {
#      "MLFLOW": {
#         "transport": "stdio",
#         # "command": "C:\\Users\\adfa\\.local\\bin\\uv.exe",
#         "command": "uv",
#         "args": [
#             "run",
#             "fastmcp",
#             "run",
#             "main.py"
#        ]
# }
# }
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MAIN_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "main.py"))


# result = subprocess.run(
#     ["fastmcp", "run", MAIN_FILE],
#     cwd=BASE_DIR,
#     capture_output=True,
#     text=True,
# )

# st.code(result.stdout)
# st.code(result.stderr)
# st.write(google.protobuf.__version__)
# FASTMCP_BIN = os.path.join(os.path.dirname(sys.executable), "fastmcp")

# SERVERS = {
#     "MLFLOW": {
#         "transport": "stdio",
#         "command": "fastmcp",
#         "args": ["run", MAIN_FILE],
#         "cwd": BASE_DIR,   # ensures main.py is found regardless of Streamlit's cwd
#     }
# }

# def run_async(coro):
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     return loop.run_until_complete(coro)

import traceback

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    try:
        return loop.run_until_complete(coro)
    except ExceptionGroup as eg:
        st.error("🚨 MCP Server crashed on startup! Check your terminal for the real error.")
        print("\n" + "="*40)
        print("🚨 REAL CAUSE OF THE CRASH 🚨")
        print("="*40)
        # Unpack and print the hidden sub-exceptions
        for exc in eg.exceptions:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        print("="*40 + "\n")
        raise



st.set_page_config(page_title="MCP Chat", page_icon="🧰", layout="centered")
st.title("🧰 MCP Chat")



load_dotenv()

# def click():
#     st.session_state.page= "list of comments"
#     st.rerun()


# b = st.button(label="hello", on_click=click())

# One-time init
if "initialized" not in st.session_state:  

    # uri = mlflow.get_tracking_uri()

    # mlflow.set_tracking_uri(uri)

    # client = MlflowClient()

    # p = client.search_experiments()

    # blobs = st.session_state.client.get_resources("MLFLOW", uris=["mlflow://experiments/all"])

    # experiment_list = blobs[0].as_string()

    assistant = MLFlowAssistant()
    run_async(assistant.initialize())
    experiment_list = run_async(assistant.get_experiments())
    st.session_state.assistant = assistant

    SYSTEM_PROMPT = (
                    "You are a helpful assistant with access to MLflow tools.\n"
                    "LIST OF EXPERIMENT:\n"
                    f"{experiment_list}\n"
                    "RULES:\n"
                    "- NEVER explain what you are about to do\n"
                    "- NEVER narrate tool usage\n"
                    "- If you need information, call the appropriate tool silently\n"
                    "- Summarize the result you get after the tool calls like a normal conversation\n"
                    "- If you cannot do something with the available tools, say: 'I don't know how to do that'\n"
                    )
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]

    
    st.session_state.initialized = True


for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        # Skip assistant messages that contain tool_calls (intermediate “fetching…”)
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # ToolMessage and SystemMessage are not rendered as bubbles


user_text = st.chat_input("Ask a question…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    # First pass: let the model decide whether to call tools
    st.session_state.history = run_async(st.session_state.assistant.ask(st.session_state.history))


        
    with st.chat_message("assistant"):
        st.markdown(st.session_state.history[-1].content or "")
    