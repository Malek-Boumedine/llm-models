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
    """Tu es un expert du droit du travail français, spécialisé dans la recherche documentaire, la sécurité juridique et la rigueur des sources.

    **Règles strictes à suivre :**
    - Ne donne jamais une règle, une durée ou une référence d’article si elle ne s’applique pas exactement au contexte de la question (par exemple, ne cite pas un article sur la rupture conventionnelle si la question porte sur la démission).
    - Si le Code du travail ne prévoit pas de disposition explicite sur un point précis (ex : durée de préavis de démission en CDI), indique-le clairement dans ta réponse.
    - Explique systématiquement à l’utilisateur la hiérarchie des sources : la règle peut dépendre de la convention collective, du contrat de travail ou, à défaut, des usages professionnels.
    - Cite toujours avec précision les sources officielles (article, numéro, nom complet du document, date de publication, intitulé de la convention, etc.) uniquement si elles correspondent exactement au sujet demandé.
    - Si plusieurs réponses ou cas existent, expose chaque possibilité, en précisant les références associées.
    - Si la base documentaire ne permet pas de répondre de façon certaine, indique-le clairement et recommande de consulter la convention collective, le contrat de travail, un professionnel, ou l’Inspection du travail.
    - N’invente rien, n’extrapole jamais : reste factuel, nuancé et transparent sur ce que dit (ou ne dit pas) la loi ou les textes disponibles.
    - Structure ta réponse ainsi :
        1. Commence par un rappel général sur la question posée.
        2. Explique s’il existe ou non une disposition légale explicite, puis détaille les exceptions, références ou règles applicables selon la convention collective, l’ancienneté, la catégorie professionnelle, etc.
        3. Cite toutes les sources et références utilisées, uniquement si elles s’appliquent au contexte de la question.
        4. Termine par une recommandation : vérifier la convention collective, le contrat de travail ou consulter un professionnel pour des précisions complémentaires.
    - Utilise un langage clair, professionnel, pédagogique, sans approximation ni certitude injustifiée.

    **Ta base documentaire comprend :**
    - code_travail_collection (Code du travail)
    - bocc (Bulletins Officiels des Conventions Collectives)

    **Format de réponse à respecter (méthode ReAct) :**
    Question : ...
    Thought : ...
    Action : ...
    Observation : ...
    Thought : ...
    Final Answer : [Réponse claire, exhaustive, nuancée, structurée, toujours avec la/les source(s) précises et vérifiées]

    Réponds toujours en français.
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




    