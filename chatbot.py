
import streamlit as st
import os

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Smart Chatbot with LangChain")

if "GROQ_API_KEY" not in os.environ:
    st.error("No se encontró la variable de entorno GROQ_API_KEY")
    st.stop()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

SESSION_ID = "streamlit_user"

user_input = st.text_input("You:", "")

if user_input:
    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": SESSION_ID}}
    ).content

    st.text_area("Chatbot:", response, height=200)
