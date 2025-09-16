import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from vector import lib
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
#logging.basicConfig(level=logging.DEBUG,filename='log.log',filemode='a',
#                    format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="HrOne Bot", page_icon="🤖")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
st.markdown("<h1 style='text-align: center;'>🦜🔗 HrOne Bot</h1>", unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
query =st.chat_input("Enter : ",key="unique_query_input")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    logging.info("User entered a question")
    results=lib.similarity_search(query,k=5)
    score= lib.similarity_search_with_score(query, k=5)
    llm=ChatGroq(
        model="groq/compound",
        api_key=os.getenv("API_KEY"),
        temperature=0
    )
    parser=StrOutputParser()
    if(score[0][1]<0.7):
        temp=""
        for doc in results:
            temp+=doc.page_content
        msg=ChatPromptTemplate.from_messages([
            ("system","u are an helpful assistant who will only answer the queries from the content given and respond as i cant answer out of my scope for other questions"),
            MessagesPlaceholder(variable_name="messages"),("user","Give a detailed explanation on the topic {query} in less than 3 paragraphs based on info from {temp} Answer:"),
        ])
        chain=msg|llm|parser
        res=chain.invoke({"query":query,"temp":temp,"messages": st.session_state.chat_history})
        with st.chat_message("assistant"):
            st.markdown(res)
        logging.info("Answered from document provided")
    else:
        msg=ChatPromptTemplate.from_messages([
            ("system","u are an helpful cheerful assistant with emojis only answer to formalities and do small talk and strictly reply i dont know for any other answers"),
             MessagesPlaceholder("messages"),
            ("user","respond to this {query}"), 
            ])
        chain=msg|llm|parser
        res=chain.invoke({"query":query,"messages": st.session_state.chat_history})
        with st.chat_message("assistant"):
            st.markdown(res)
        logging.info("Assistant answered")
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=res))  
    logging.info("History appended")  
