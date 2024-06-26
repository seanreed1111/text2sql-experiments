from pathlib import Path
import ast 
import os
import json
import streamlit as st
import pandas as pd
from langchain_openai import AzureChatOpenAI
# from langchain_community.chat_models.azure_openai import AzureChatOpenAI #deprecated class, fix later
# from langchain.agents import create_sql_agent
# from langchain.sql_database import SQLDatabase 
from langchain_community.utilities.sql_database import SQLDatabase #using sqlalchemy
# from langchain.callbacks import StreamlitCallbackHandler
# from langchain_community.callbacks import StreamlitCallbackHandler
# import sqlalchemy
# from sqlalchemy import create_engine
import urllib
# from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from loguru import logger

# streamlit examples: https://streamlit.io/generative-ai
# LANGCHAIN_PROJECT = "experiment-6-chat-with-sql-no-agent"
LANGCHAIN_PROJECT = Path(__file__).stem
st.set_page_config(
    page_title=LANGCHAIN_PROJECT,
    page_icon="🌄"
)
config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"
MAX_TOKENS = 1000

# add separate memory thread for each tab
tab_titles=[
    "LLM Only",
    "SQL Connection w/o LLM",
    "LLM + SQL Execution Agent",
    "LLM + Python Agent",
    "LLM + Visualization Agent"
]
# tabs = st.tabs(tab_titles)
# tab_dict = {tab_title:tab_handle for tab_title, tab_handle in zip(tab_titles,tabs)}

llm_tab, sql_tab, sql_agent_tab, python_agent_tab, visualization_tab  = st.tabs(tab_titles)
tab_handles = [llm_tab, sql_tab, sql_agent_tab, python_agent_tab, visualization_tab]

tab_dict = {tab_title:tab_handle for tab_title, tab_handle in zip(tab_titles,tab_handles)}

#set title at top of tab 
for tab_heading, tab_handle in zip(tab_titles, tab_handles):
    tab_handle.title(tab_heading)
    
   
def load_schema_from_file(file, dir = config_dir_path):
    try:
        file_path = dir / file
        with open(file_path, 'r') as f:
            schema = f.read()
        assert schema is not None
        return schema
    
    except Exception as e:
        logger.error(e)

uploaded_schema = load_schema_from_file(file="DDL_for_LLM_upload.sql")
def run_azure_config(config_dir = config_dir_path):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

#### Need get_db_engine to run asynchronously so it doesn't block loading the rest of page
# https://docs.python.org/3/library/asyncio.html
## jason brownlee https://superfastpython.com/python-async-function/
# https://superfastpython.com/asyncio-run-program/
# https://medium.com/@danielwume/an-in-depth-guide-to-asyncio-and-await-in-python-059c3ecc9d96
# https://docs.python.org/3/library/asyncio-task.html
####

@logger.catch
@st.cache_resource(ttl="4h")
def get_db_engine(db_config_file="dbconfig.json", config_dir_path = config_dir_path, **kwargs):
    
    if not kwargs: 
        kwargs = {"schema":"sandbox"}
    
    @st.cache_resource(ttl="4h")    
    def get_wab_connection_string(db_config_file=db_config_file, config_dir_path=config_dir_path ):
        driver= '{ODBC Driver 18 for SQL Server}'
        db_config_path = config_dir_path / db_config_file

        with open(db_config_path) as json_file:
            dbconfig = json.load(json_file)

        server = dbconfig['server']
        database = dbconfig['database']
        uid = dbconfig['username']
        pwd = dbconfig['password']
        port = int(dbconfig.get("port",1433))
        pyodbc_connection_string = f"DRIVER={driver};SERVER={server};PORT={port};DATABASE={database};UID={uid};PWD={pwd};Encrypt=yes;Connection Timeout=30;READONLY=True;"
        params = urllib.parse.quote_plus(pyodbc_connection_string)
        sqlalchemy_connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        return sqlalchemy_connection_string
    

    return SQLDatabase.from_uri(database_uri=get_wab_connection_string(),
                                **kwargs
                               )

# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text=""):
#         self.container = container
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.text += token
#         self.container.markdown(self.text)

run_azure_config()

with sql_tab:
    db = get_db_engine()
    q = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'sandbox'
    ORDER BY TABLE_NAME;
    """

    with st.expander("loading tab"):
        try:
            db.run(q)
            st.success('Sucessfully connected to the database')
        except Exception as e:
            st.error(e)

    # have user enter their query
    # execute query in SQLDatabase
    # return results inside a pandas dataframe
    query = st.text_input(label="Enter a SQL query", value="select top 2 *FROM sandbox.ACCOUNT_CATEGORY_TYPE")

    # test to see if can parse the response from the database
    if query:
        try:
            results = db.run(query)
            # st.text(results)
            # st.text(f"type of results is:{type(results)}") #DB returns a string representation. need to make into json
            python_obj_from_results = ast.literal_eval(results)
            # st.text(f"type of python_obj_from_results is:{type(python_obj_from_results)}") #type:list
            # write a function that lets you return the column names to put on top of results

            regular_df = pd.DataFrame(python_obj_from_results)
            st.dataframe(regular_df, use_container_width=True)
            # st.data_editor(regular_df, use_container_width=True) #editable dataframe
        except Exception as e:
            logger.error(e)
            st.error(e)

# def reset_llm_only_tab():
#     st.session_state["reset_llm_only_tab"] = True

# start thinking about chat memory. maybe keep system message plus last N items
with llm_tab:
    SCHEMA_FILEPATH  = config_dir_path / "DDL_for_LLM_upload_sample.sql"
    uploaded_schema = f"{load_schema_from_file(SCHEMA_FILEPATH)}"
    reset_button = st.button("Reset Chat")

    if reset_button or ("messages" not in st.session_state):
        st.session_state["messages"] = None
        
        system_message = f"""
        You are an expert at writing Mircosoft SQL database queries and T-SQL code. 
        When asked to write SQL queries use the following schema
        \n\n\n
        {uploaded_schema}
        \n\n\n
        After writing a query, score its estimated accuracy in answering the 
        user's question on a scale of 1-5, with 1 the lowest score and 5 the 
        highest possible. Respond with the query and the accuracy score. If you give
        an accuracy score of 1 or 2, briefly state your reason.
        """
        # st.session_state["system_message"] = ChatMessage(role="system", content=system_message)
        st.session_state["messages"] = [ChatMessage(role="system", content=system_message), 
                                        ChatMessage(role="assistant", content="How can I help you?")
                                        ]

    for msg in st.session_state.messages:
        if msg.role != "system":
            st.chat_message(msg.role).write(msg.content)

    # what is the difference between chat_input and text_input?
    if prompt := st.chat_input():  #prompt written on top
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)


        with st.chat_message("assistant"):
            llm = AzureChatOpenAI(
                temperature=0,
                streaming=False,
                max_tokens=MAX_TOKENS,
                azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
                model_name=os.environ["MODEL_NAME_GPT35"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                request_timeout=45,
                verbose=True,
            )
            response = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))