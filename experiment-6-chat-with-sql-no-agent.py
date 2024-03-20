from pathlib import Path
import os
import json
import streamlit as st
from langchain_openai import AzureChatOpenAI
# from langchain_community.chat_models.azure_openai import AzureChatOpenAI #deprecated class, fix later
# from langchain.agents import create_sql_agent
# from langchain.sql_database import SQLDatabase 
from langchain_community.utilities.sql_database import SQLDatabase #using sqlalchemy
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler
import sqlalchemy
from sqlalchemy import create_engine
import urllib
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "experiment-6-chat-with-sql-no-agent"
st.set_page_config(
    page_title=LANGCHAIN_PROJECT,
    page_icon="🌄"
)
config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"

# add separate memory thread for each tab
tab_titles=[
    "SQL Connection w/o LLM",
    "LLM Only",
    "LLM + SQL Execution Agent",
    "LLM + Python Agent",
    "LLM + Visualization Agent"
]
sql_tab, llm_tab, sql_agent_tab, python_agent_tab, visualization_tab  = st.tabs(tab_titles)
tab_handles = [sql_tab, llm_tab, sql_agent_tab, python_agent_tab, visualization_tab]

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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

run_azure_config()

with llm_tab:
    SCHEMA_FILEPATH  = config_dir_path / "DDL_for_LLM_upload_sample.sql"
    uploaded_schema = f"{load_schema_from_file(SCHEMA_FILEPATH)}"

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

def reset_llm_only_tab():
    st.session_state["reset_llm_only_tab"] = True
    
with llm_tab:
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

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)


        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = AzureChatOpenAI(
                temperature=0,
                streaming=True,
                max_tokens=800,
                azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
                model_name=os.environ["MODEL_NAME_GPT35"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                request_timeout=45,
                verbose=True,
                callbacks=[stream_handler]
            )
            response = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))