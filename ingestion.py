import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import spacy
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

print(os.getcwd())

chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")))
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

loader = PyPDFLoader('IJCSE-V10I6P101.pdf')
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(document)

vectorstore = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function,
    collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    client=chroma_client,
)
print(f"Added {len(chunked_documents)} chunks to chroma db")
