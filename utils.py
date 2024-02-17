from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import os
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version = os.getenv("OPENAI_API_VERSION"),
    deployment_name = os.getenv("AZURE_GPT_DEPLOYMENT_NAME"),
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_type = "azure",
    temperature = 0
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_GPT_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_type = "azure",
)

retriever = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=embeddings
).as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}

Question: {question}

""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)