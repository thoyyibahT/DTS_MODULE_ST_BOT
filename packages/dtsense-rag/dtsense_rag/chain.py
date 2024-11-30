import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get Groq API key
#groq_api_key = os.environ['GROQ_API_KEY']
model = 'llama-3.2-11b-vision-preview'
# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'], 
        model_name=model
)
# response = groq_chat.invoke("Who are the authors of DTSense Streamlit Book?")
# print(response)

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.
Please answer with the same language as the question.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# basic_chain = prompt | groq_chat | StrOutputParser()
# response = basic_chain.invoke({"context": "", "question": "Who are the authors of DTSense Streamlit Book?"})
# print(response)

embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

URI = "packages/dtsense-rag/dtsense_rag/data/milvus_lite.db"

# vectordb = Milvus(
#     embedding_function=HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
#     ),
#     collection_name="dtsense_streamlit",
#     connection_args={"uri": URI},
# )
# retriever = vectordb.as_retriever()

from langchain.vectorstores import FAISS

URI_FAISS = "packages/dtsense-rag/dtsense_rag/data/"
faiss_index = FAISS.load_local(URI_FAISS, embedding, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()

# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | groq_chat
    | StrOutputParser()
)

# response = rag_chain.invoke("Siapa dari the authors dari module DTSense Streamlit?")
# print(response)