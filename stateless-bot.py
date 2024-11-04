import os
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)
import chromadb
import chromadb.api
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("This RAG was given 2 pdf file, Metamorphosis by Kafka and a Paper on Generative AI, so ask question based on it")

chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")))
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name=os.getenv("CHROMA_COLLECTION_NAME"))

model = OllamaLLM(model="llama3")

template = """
Answer the question below

Here is the conversation history: {context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

context = ""
while True:
    query = input("User : ")
    results = collection.query(
    query_texts=[query],
    n_results=2
    )
    retrieved_documents = [doc for doc in results['documents']]
    rd = " ".join(retrieved_documents[0])
    if(query.lower() == "exit"): break
    result = chain.invoke({"question":query,"context" : context})
    print("Bot: " +result)
    context += f"\nSummary:{rd}\nUser: {query}\nAI: {result}"