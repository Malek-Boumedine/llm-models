import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 

import warnings
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")

load_dotenv()



# =================================================================================

# agent droit du travail

qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/droit_travail_agent"

agent_type = "droit_du_travail"
domains = ["droit du travail français"]
speciality = "droit_du_travail"
description = "Agent spécialisé du droit du travail français"

collections = ["code_travail_collection", "bocc"]
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert du droit du travail français spécialisé dans la sécurité juridique.

    **Règles strictes :**
    - Citations exactes d'articles uniquement si applicables
    - Indique clairement les limites légales et renvois aux conventions collectives
    - Hiérarchie : loi → convention → contrat → usages
    - Transparence sur les incertitudes
    - Recommandations pour approfondir

    **Format de réponse :**
    - Réponse directe et structurée
    - Sources avec articles précis
    - Degré de certitude indiqué
    - Recommandations si nécessaire

    Réponds en français avec un ton professionnel et rigoureux.
    """),
    ("placeholder", "{messages}"),
])


class DroitTravailAgent(BaseAgent) :
    def __init__(self, 
        qdrant_host = qdrant_host, 
        embedding_model = embedding_model, 
        log_file_path = log_file_path, 
        model_name = model_name, 
        agent_type = agent_type, 
        domains = domains, 
        speciality = speciality, 
        description = description, 
        collections = collections, 
        prompt = prompt):
        super().__init__(qdrant_host, embedding_model, log_file_path, model_name, agent_type, domains, speciality, description, collections, prompt)


# =================================================================================

# TEST 

if __name__ == "__main__" : 
    
    agent_conventions_collectives = DroitTravailAgent()

    # result = agent_conventions_collectives.query("Quel est le délai de préavis pour la démission d'un salarié en CDI ?")["response"]
    result = agent_conventions_collectives.query("Un salarié peut-il être licencié pour faute grave en cas d'absence injustifiée pendant plusieurs jours ?")["response"]
    print(result)




    