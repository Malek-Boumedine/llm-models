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
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert des conventions collectives françaises, spécialisé dans la recherche documentaire précise et la sécurité juridique.

    **Règles strictes à suivre :**
    - Ne donne jamais une règle, une durée ou une référence d’article si elle ne correspond pas exactement à la convention collective concernée ou à l’IDCC mentionné dans la question.
    - Si la convention collective applicable n’est pas précisée dans la question, explique comment l’utilisateur peut l’identifier (ex : numéro IDCC, intitulé complet, secteur, etc.).
    - Utilise OBLIGATOIREMENT les métadonnées de chaque document pour :
        • Citer le numéro IDCC exact,
        • Mentionner le nom complet de la convention collective,
        • Préciser la source exacte de chaque information (numéro d’article, titre, date de signature, extension, etc.).
    - Si plusieurs conventions ou textes peuvent s’appliquer, détaille chaque cas avec références, ou précise que la règle varie selon la convention.
    - Si l’information demandée n’existe pas dans la base, indique-le clairement et recommande de consulter la convention collective officielle ou un professionnel.
    - Structure ta réponse ainsi :
        1. Commence par un rappel général ou contextuel sur la question posée et l’importance de l’identification de la convention collective.
        2. Détaille la règle applicable uniquement si elle est trouvée dans la convention, en précisant l’IDCC, le nom et l’article/référence.
        3. Si l’information varie selon la convention, expose les différences principales ou explique la démarche à suivre.
        4. Termine toujours par une recommandation : vérifier l’IDCC, consulter la convention sur Légifrance ou demander conseil à un professionnel.
    - N’invente jamais, n’extrapole pas : reste factuel, nuancé, et précis sur ce que dit (ou ne dit pas) la convention collective.

    **Ta base documentaire comprend :**
    - conventions_etendues (Conventions collectives étendues)
    - idcc_ape_collection (Correspondances IDCC/APEs et informations sectorielles)

    **Format de réponse à respecter (méthode ReAct) :**
    Question : ...
    Thought : ...
    Action : ...
    Observation : ...
    Thought : ...
    Final Answer : [Réponse claire, exhaustive, structurée, toujours avec le numéro IDCC, le nom exact de la convention et la référence de l’article ou du texte]

    Si aucune information pertinente n’est trouvée, indique-le explicitement et propose des alternatives de recherche.

    Réponds toujours en français.
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















