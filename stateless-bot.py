import os
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,)
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
import chromadb.api
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")


load_dotenv()

chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT")))
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


collection = chroma_client.get_collection(name=os.getenv("CHROMA_COLLECTION_NAME"))
results = collection.query(
    query_texts=["what loss functions are used "],
    n_results=1
)


retrieved_documents = [doc for doc in results['documents']]
# retrieved_documents = '.'.join(retrieved_documents[0])
# print(retrieved_documents)

# Create a Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    # max_length=1024, 
    max_new_tokens=100,
    do_sample=True,
    temperature=0.2
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Use the pulled prompt in a chain
prompt_template = PromptTemplate(input_variables=['context', 'question'], template="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")


question = "What are the applications of generative AI?"
retrieved_documents = '.'.join(retrieved_documents[0])

# print(retrieved_documents)

chain = LLMChain(prompt=prompt_template, llm=llm)
response = chain.run({
    'context': retrieved_documents,
    'question': question
})

file_name = "output.txt"
with open(file_name, 'w') as file:
    file.write(response)

print(f"Written to {file_name}")