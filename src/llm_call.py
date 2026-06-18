# mcp_client.py

import asyncio
import json
import os
import shutil
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# UV = shutil.which("uv")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_FILE = os.path.join(BASE_DIR, "main.py")

SERVERS = {
    "MLFLOW": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            MAIN_FILE
        ]
    }
}


class MLFlowAssistant:
    def __init__(self):
        self.client = MultiServerMCPClient(SERVERS)
        self.tools = None
        self.named_tools = None
        self.llm_with_tools = None

    async def initialize(self):
        self.tools = await self.client.get_tools()
        self.named_tools = {t.name: t for t in self.tools}
        self.llm_with_tools = llm.bind_tools(self.tools)

    async def get_experiments(self):
        blobs = await self.client.get_resources("MLFLOW", uris=["mlflow://experiments/all"])
        return blobs[0].as_string()
        

    async def ask(self, messages):
        first = await self.llm_with_tools.ainvoke(messages)
        messages.append(first)

        if not first.tool_calls:
            return messages

        for tc in first.tool_calls:
            result = await self.named_tools[tc["name"]].ainvoke(tc.get("args", {}))
            messages.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content=json.dumps(result)
                )
            )

        final = await self.llm_with_tools.ainvoke(messages)
        messages.append(final)

        return messages