from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
import warnings
import os
# warnings.filterwarnings("ignore")

load_dotenv()



# =================================================================================

qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/conventions_collectives_agent"

agent_type = "conventions_collectives"
domains = ["conventions collectives du droit français"]
speciality = "conventions_collectives"
description = "Agent spécialisé dans les conventions collectives du droit du travail français"

collections = ["conventions_etendues"]
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert des conventions collectives françaises spécialisé dans la précision juridique.

    **Règles strictes :**
    - Citations exactes avec IDCC, nom complet et article précis
    - Utilise les métadonnées pour référencer (IDCC, date, extension)
    - Si convention non spécifiée, explique comment l'identifier
    - Transparence sur les limites et variations
    - Recommandations pour approfondir

    **Structure de réponse :**
    1. Contexte et importance de l'identification
    2. Règle applicable avec références exactes
    3. Variations selon les conventions si applicable
    4. Recommandations (vérifier IDCC, consulter Légifrance)

    **Collection disponible :**
    - conventions_etendues (Conventions collectives étendues)

    Réponds en français avec un ton professionnel et rigoureux.
    """),
    ("placeholder", "{messages}"),
])


class ConventionsCollectivesAgent(BaseAgent) :
    
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
    prompt = prompt) :
        super().__init__(qdrant_host, embedding_model, log_file_path, model_name, agent_type, domains, speciality, description, collections, prompt)
        

# =================================================================================
        
# TEST 

if __name__ == "__main__" : 
    
    agent_conventions_collectives = ConventionsCollectivesAgent()

    result = agent_conventions_collectives.query("Quel est le montant de la prime d’ancienneté prévue par la Convention collective nationale du commerce de détail et de gros à prédominance alimentaire ?")["response"]
    print(result)

    # # Vérification des capacités
    # capabilities = agent_conventions_collectives.get_capabilities()
    # print(capabilities)















