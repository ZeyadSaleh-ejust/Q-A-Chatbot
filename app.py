import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with groq"


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question: {question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens): # "Llama3-8b-8192"
    '''
    temperature -> if equal 0 --> it's mean the model will not be creative with respect to answers 
    as if I ask the same question it will responed the same answer
    temperature -> if equal 1 --> more creative response
    ''' 
    groq_api_key = api_key
    llm = ChatGroq(model_name = llm,groq_api_key=groq_api_key,temperature = temperature,max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enchaned Q&A Chatbot with groq")
api_key = st.sidebar.text_input("Enter your groq API Key:",type="password")

## Drop down to select various groq models
llm = st.sidebar.selectbox("Select an groq AI Model",["Llama3-8b-8192","gemma2-9b-it","llama3-70b-8192"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value =0.0,max_value = 1.0,value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value =50,max_value = 300,value = 150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the groq API key In the sider bar")
else:
    st.write("Please provide the question")
    

