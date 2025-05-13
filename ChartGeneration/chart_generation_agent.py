# Chart Generation Agent with DuckDB

import os
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = ""

# DuckDB Loader Example
from langchain_community.document_loaders import DuckDBLoader

# Build the Chart Generation Agent
from typing_extensions import TypedDict
# from langchain_core.stores import InMemoryStore
# from langgraph.store.memory import InMemoryStore
from langchain.storage import InMemoryStore
global store
store = InMemoryStore()
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    code: str

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

from langchain import hub
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
# assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()

# Define Write Query
from typing_extensions import Annotated
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    file_path = "data/AB_NYC_2019.csv"
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": "duckdb",
            "top_k": 10,
            "table_info": f"read_csv_auto('{file_path}')",
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    data = DuckDBLoader(state["query"]).load()
    return {'result': data}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def write_code_for_chart(state: State):
    """Generate Python Code for Data Visualization """
    chart_type = store.mget(['chart_type'])[0]
    prompt = (
        "Given the following user question, corresponding SQL query, "
        f"and SQL result, build a python plotly script to with {chart_type} chart to show the data, return executable code only\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"code": response.content}

def execute_python_code(state: State):
    code = state["code"]
    code = code.replace("python", "")
    code = code.replace("```", "")
    exec(code)

from langgraph.graph import START, StateGraph
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer, write_code_for_chart, execute_python_code]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

from IPython.display import Image, display
#display(Image(graph.get_graph().draw_mermaid_png()))

#from langchain.storage import InMemoryStore
def llm_chart(question: str, chart_type: str):
    store = InMemoryStore()
    store.mset([('chart_type', chart_type)])
    for step in graph.stream({"question": question}, stream_mode="updates"):
        print(step)

question = "This is a dataset about New York Airbnb, can you provide average price for each neighbourhood and rank them in descending order? "
chart_type = 'pie'
llm_chart(question, chart_type)

question = "This is a dataset about New York Airbnb, can you show all the host by latitude and longitude"
chart_type = 'map'
llm_chart(question, chart_type)