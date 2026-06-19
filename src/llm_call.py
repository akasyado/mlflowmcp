import asyncio
import json
import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash",
#             temperature=0,
#             max_retries=2
#         )




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
        self.llm = llm


    # async def llm_call(self, user_query):
    #     system_message = (
    # "You are a query-refinement assistant. Your job is to read the user's request and "
    # "rewrite it into a clear, precise instruction that a downstream LLM (equipped with the "
    # "tools listed below) can act on correctly. Do not answer the query yourself — only refine it.\n\n"

    # "Guidelines for refining:\n"
    # "- Resolve ambiguity (e.g. vague experiment/model references) by stating exactly what "
    # "information is needed to resolve it (e.g. 'first look up the experiment id via list_experiments').\n"
    # "- If the user refers to an experiment or model by name, explicitly note that the id must be "
    # "looked up first, since tools require ids, not names.\n"
    # "- Preserve the user's intent and all relevant details (ids, versions, run names, metrics) "
    # "without adding assumptions.\n"
    # "- Make the refined query a single, self-contained instruction.\n\n"

    # "Tools available to the downstream LLM:\n\n"

    # "1. runs_in_experiment(experiment_id: list)\n"
    # "   - Given a list of experiment ids, returns all runs for each experiment, including each "
    # "run's id, parameters, metrics, and tags.\n"
    # "   - Requires experiment ids, NOT experiment names. If the user gives a name, the name must "
    # "first be resolved to an id (e.g. via list_experiments).\n\n"

    # "2. models_in_production()\n"
    # "   - Lists every model version currently in production, including run id, version, "
    # "experiment name, metrics, and parameters. Takes no arguments.\n\n"

    # "3. push_model_to_production(run_id: list)\n"
    # "   - Registers the given run id(s) as new model version(s) in production.\n"
    # "   - Requires a list of run ids (not run names or experiment ids).\n\n"

    # "4. remove_model_from_production(version: str)\n"
    # "   - Removes the specified version from production.\n"
    # "   - Requires the model version string, not a run id.\n\n"

    # "Return only the refined query — no explanation, no extra commentary."
    # )
    #     input =[SystemMessage(content=system_message), HumanMessage(content=user_query)]
    #     result = await self.llm.ainvoke(input)
    #     return result.content
    async def get_experiments(self):
        blobs = await self.client.get_resources("MLFLOW", uris=["mlflow://experiments/all"])
        return blobs[0].as_string()
        

    async def ask(self, messages):
        first = await self.llm_with_tools.ainvoke(messages)
        messages.append(first)

        if not first.tool_calls:
            return messages

        for i ,tc in enumerate(first.tool_calls):
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


    # async def ask(self, messages, on_token=None):
    #     """
    #     on_token: optional callback(str) called with each new text token as it streams in.
    #     """
    #     first = await self._stream_and_accumulate(messages[-5:], on_token)
    #     messages.append(first)

    #     if not first.tool_calls:
    #         return messages

    #     for i, tc in enumerate(first.tool_calls):
    #         result = await self.named_tools[tc["name"]].ainvoke(tc.get("args", {}))
    #         messages.append(
    #             ToolMessage(
    #                 tool_call_id=tc["id"],
    #                 content=json.dumps(result)
    #             )
    #         )

    #     final = await self._stream_and_accumulate(messages[-(5+i):], on_token)
    #     messages.append(final)

    #     return messages

    # async def _stream_and_accumulate(self, messages, on_token=None):
    #     """Streams a response and returns the fully-merged AIMessage chunk."""
    #     full = None
    #     async for chunk in self.llm_with_tools.astream(messages):
    #         full = chunk if full is None else full + chunk
    #         if on_token and chunk.content:
    #             on_token(chunk.content)
    #     return full
    

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    assistant =  MLFlowAssistant()
    asyncio.run(assistant.initialize())
    experiment_list = asyncio.run(assistant.get_experiments())

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
    msg = [SystemMessage(content=SYSTEM_PROMPT)]
    msg.append(HumanMessage(content="What are the artifacts of the run 775a2cba640048b994eb74dbb9830ad4"))
    msg = asyncio.run(assistant.ask(msg))
    print(msg)