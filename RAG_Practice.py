from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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
    if(query.lower() == "exit"): break
    result = chain.invoke({"question":query,"context" : ""})
    print("Bot: " +result)
    context += f"\nUser: {query}\nAI: {result}"