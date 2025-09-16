from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=ChatGroq(
    model="groq/compound",
    api_key=os.getenv("API_KEY")
)
#prompt=PromptTemplate.from_template("Answer in less than 5 lines:{question}")
parser = StrOutputParser()
msg=ChatPromptTemplate.from_messages([
    ("system","u are an helpful assistant use emojis be cheerful "
    "asnwer nly AI related questions and reply I Dont Know strictly dont answer to other domains Answer in less than 5 lines:{question}"),
    ("user","{question}")
])
while True: 
    user=input("You: ")
    chain=msg|llm| parser
    res=chain.invoke({"question":user})
    print(res)