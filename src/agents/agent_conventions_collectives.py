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
log_file_path = "logs/droit_travail_agent"
log_file_path = "logs/conventions_collectives_agent"

agent_type = "conventions_collectives"
domains = ["conventions collectives du droit français"]
speciality = "conventions_collectives"
description = "Agent spécialisé dans les conventions collectives du droit du travail français"

collections = ["conventions_etendues","idcc_ape_collection"]
# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#     """Tu es un expert dans les conventions collectives du droit du travail français.
#     Tu réponds toujours en t'appuyant exclusivement sur les informations contenues dans ta base documentaire.
        
#     Tes collections disponibles sont :
#     - conventions_etendues
#     - idcc_ape_collection
    
#     IMPORTANT : Utilise OBLIGATOIREMENT les métadonnées des documents pour :
#     - Citer le numéro IDCC exact 
#     - Mentionner le nom de la convention 
#     - Préciser la source exact de chaque information
        
#     Utilise la méthode ReAct suivante :
#     Question: la question initiale de l'utilisateur
#     Thought: explique ta réflexion sur comment trouver la réponse
#     Action: utilise l'outil correspondant à la collection la plus pertinente
#     Observation: les résultats de l'outil
#     Thought: ta réflexion suite aux résultats
#     Final Answer: réponds à l'utilisateur de manière claire, détaillée et structurée en français.

#     Si aucune information n'est trouvée, indique-le clairement.
#     """),
#     ("placeholder", "{messages}"),])

prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un expert dans les conventions collectives du droit du travail français.
    Tu réponds en utilisant les informations de ta base documentaire.
    
    Utilise tes outils de recherche disponibles pour trouver les informations. cherche bien dans les metadata aussi des collections.
    Réponds clairement et cite les sources quand possible.
    """),
    ("placeholder", "{messages}"),
])


class ConventionsCollectivesAgent(BaseAgent) :
    
    def __init__(self, qdrant_host, embedding_model, log_file_path ,model_name, agent_type, domains, speciality, description, collections, prompt) :
        super().__init__(qdrant_host, embedding_model, log_file_path, model_name, agent_type, domains, speciality, description, collections, prompt)
        

# =================================================================================
        
# TEST 

if __name__ == "__main__" : 
    
    agent_conventions_collectives = ConventionsCollectivesAgent(
        qdrant_host = qdrant_host, 
        embedding_model = embedding_model, 
        model_name = model_name, 
        log_file_path = log_file_path, 
        agent_type = agent_type, 
        domains = domains, 
        speciality = speciality, 
        description = description, 
        collections = collections, 
        prompt = prompt
    )

    result = agent_conventions_collectives.query("Quel est le salaire minimum conventionnel brut menseul selon la Convention collective nationale de travail du personnel des organismes de sécurité sociale ?")["response"]
    print(result)

    # # Vérification des capacités
    # capabilities = agent_conventions_collectives.get_capabilities()
    # print(capabilities)















