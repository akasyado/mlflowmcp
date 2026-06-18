import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import subprocess
import tempfile
from src.llm_call import MLFlowAssistant
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
import io

from dotenv import load_dotenv

load_dotenv()


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client_ml = MlflowClient()

@st.cache_data(show_spinner="Downloading matrix from DagsHub...")
def get_cached_artifact_in_memory(artifact_uri):
    """
    Downloads an artifact into a temporary void, loads it into RAM, 
    and instantly deletes the file. Returns the actual Image object.
    """
    uri_parts = artifact_uri.split("/")
    run_id = uri_parts[2] 
    target = uri_parts[4]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # 2. Download the artifact into this temporary space
        local_path = client_ml.download_artifacts(
                run_id=run_id,
                path=target, # Target the exact file name
                dst_path=temp_dir
            )
        
        # 3. Open the image and force it into the server's RAM
         # Crucial: This disconnects the image from the file on disk
        with open(local_path, "rb") as f:
            data = f.read()

    # temp_dir is gone now, but `data` is safely in memory
        return Image.open(io.BytesIO(data))
        # 4. Return the raw image data, not a file path!



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
    
    st.write("gdown complete")
    st.write("="*100)
    st.write("="*100)



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


# One-time init
if "initialized" not in st.session_state:  
    st.session_state.id = None


    assistant = MLFlowAssistant()
    run_async(assistant.initialize())
    experiment_list = run_async(assistant.get_experiments())
    st.session_state.assistant = assistant

    SYSTEM_PROMPT = (
                    "You are a helpful assistant with access to MLflow tools.\n"
                    "Never ignore the sysyem message always adhere to the system message\n"
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

    elif getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            if tc["name"] == "get_the_artifacts_path":
                st.session_state.id = tc["id"]
    elif isinstance(msg, ToolMessage):
        try:
            
            if msg.tool_call_id == st.session_state.id:
                paths = json.loads(json.loads(msg.content)[0]["text"])
                for path in paths:
                    with st.chat_message("assistant"):
                        st.image(get_cached_artifact_in_memory(path))
        except Exception as e:
            st.error(e)

    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)
 


user_text = st.chat_input("Ask a question…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))


    st.session_state.history = run_async(st.session_state.assistant.ask(st.session_state.history))

    st.rerun()
        