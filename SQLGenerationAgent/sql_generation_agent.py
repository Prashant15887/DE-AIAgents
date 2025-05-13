# SQL Generation Agent with DuckDB
import os
import getpass

os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = ""

# DuckDB Loader Example
from langchain_community.document_loaders import DuckDBLoader

file_path = "data/netflix_titles.csv"
loader = DuckDBLoader(f"SELECT * FROM read_csv_auto('{file_path}') LIMIT 10")
data = loader.load()
print("Sample data")
print(data)

# Build the SQL Generation Agent
from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# from langchain_openai import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

from langchain import hub
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
# print("query_prompt_template")
# print(query_prompt_template)
# print("len(query_prompt_template.messages)")
# print(len(query_prompt_template.messages))
# assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()

# Define Write Query
from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
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

sql_query = write_query({"question": "How many shows are there?"})
print("SQL query for question: 'How many shows are there?'")
print(sql_query)

def execute_query(state: State):
    """Execute SQL query."""
    data = DuckDBLoader(state["query"]).load()
    return {'result': data}

output = execute_query(sql_query)
print(output)

sql_query = write_query({"question": "Can you get the total shows per director, and sort by total shows in descending order?"})
print("SQL query for question: 'Can you get the total shows per director, and sort by total shows in descending order?'")
print(sql_query)
output = execute_query(sql_query)
print(output)

sql_query = write_query({"question": "Can you get number of shows start with letter D?"})
print("SQL query for question: 'Can you get number of shows start with letter D?'")
print(sql_query)
output = execute_query(sql_query)
print(output)

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

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "Can you get the total shows per director, and sort by total shows in descending order for the top 3 director?"}, stream_mode="updates"
):
    print(step)

for step in graph.stream(
    {"question": "Can you get number of shows start with letter D?"}, stream_mode="updates"
):
    print(step)

for step in graph.stream(
    {"question": "Can you get number of shows start with letter D?"}, stream_mode="updates"
):
    print(step)

for step in graph.stream(
    {"question": "Can you get the how many years between each show director Rajiv Chilaka produced, sort by release years?"}, stream_mode="updates"
):
    print(step)