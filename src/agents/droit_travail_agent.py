import os
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 

from langchain_community.vectorstores import Chroma
import warnings
# warnings.filterwarnings("ignore")




# =================================================================================

# Connexion à Qdrant (exemple en local)
qdrant_host = "http://localhost:6333"
collection_name = "code_travail_collection"  # adapte à ta collection
embedding_model = "paraphrase-multilingual-mpnet-base-v2"
embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)

# qdrant_db = Qdrant(
#     url=qdrant_host,
#     collection_name=collection_name,
#     embeddings=embedding_function)

# # création du retriever
# retriever = qdrant_db.as_retriever(
#     search_kwargs={"k": 5}  # Nombre de passages renvoyés
# )


# =================================================================================

chroma_db = Chroma(
    persist_directory="BDD_TEST",   # ton dossier Chroma
    collection_name="code_travail_collection",
    embedding_function=embedding_function,
)

retriever = chroma_db.as_retriever(search_kwargs={"k": 5})


# =================================================================================

# Création de l'outil LangChain pour la recherche
retriever_tool = create_retriever_tool(
    retriever,
    name="recherche_code_travail",
    description="Recherche d'informations précises dans le Code du travail et les conventions collectives.",
)

tools = [retriever_tool]

# Prompt contextuel
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu te nommes Urssafino et tu es un expert en droit du travail français, tu t'appuies sur les documents officiels de la base."),
    ("placeholder", "{messages}"),
    ("user", "Sois aussi précis que possible dans tes réponses, cite les références si possible."),
])

model = ChatOllama(model="mistral:7b")
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
    
    