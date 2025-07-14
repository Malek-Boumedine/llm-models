import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 

from langchain_community.vectorstores import Chroma
import warnings
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")




# =================================================================================

# Connexion à Qdrant (exemple en local)
with open("fichier_log_agent_test", "w", encoding="utf-8") as logfile:
    qdrant_host = "http://localhost:6333"
    embedding_model = "paraphrase-multilingual-mpnet-base-v2"
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
    client = get_qdrant_client(qdrant_host, logfile)

collections = ["code_travail_collection", "conventions_etendues", "bocc", "idcc_ape_collection"]
retriever = [
    Qdrant(
        client=client,
        collection_name=col,
        embeddings=embedding_function
    ).as_retriever(search_kwargs={"k": 5})
    for col in collections]

# =================================================================================

# Création de l'outil LangChain pour la recherche
retriever_tools = [
    create_retriever_tool(
        ret,
        name=f"recherche_{col}",
        description=f"Recherche dans la collection {col}")
    for ret, col in zip(retriever, collections)]

tools = retriever_tools

# Prompt contextuel
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Tu es un expert en droit du travail français.
     Tu réponds toujours en t'appuyant exclusivement sur les informations contenues dans ta base documentaire.
     
     Tes collections disponibles sont :
     - code_travail_collection
     - conventions_etendues
     - bocc
     - idcc_ape_collection

     Utilise la méthode ReAct suivante :
     Question: la question initiale de l'utilisateur
     Thought: explique ta réflexion sur comment trouver la réponse
     Action: utilise l'outil correspondant à la collection la plus pertinente
     Observation: les résultats de l'outil
     Thought: ta réflexion suite aux résultats
     Final Answer: réponds à l'utilisateur de manière claire, détaillée et structurée en français.

     Si aucune information n'est trouvée, indique-le clairement.
     """),
    ("placeholder", "{messages}"),
    ("user", "{input}"),
])

# model = ChatOllama(model="mistral:7b")
model = ChatOllama(model="llama3.1:latest")
agent_memory = MemorySaver()

agent = create_react_agent(
    model,
    tools,
    checkpointer=agent_memory)


def print_stream(agent, question):
    config = {"configurable": {"thread_id": "abc123"}}
    result = agent.invoke({"messages": [("user", question)]}, config)
    print(result["messages"][-1].content)

# ==============================================================================================================

if __name__ == "__main__":
    
    while True :
        print("\n" + "="*60 + "\n")
        question = input("User : ")
        if question == "q" : 
            break
        print("\n" + "Agent : " + "\n")
        print_stream(
            agent,
            f"{question}"
        )
    
    