import os
from langchain_core.prompts import ChatPromptTemplate
import warnings
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")


load_dotenv()


# =================================================================================

# agent droit du travail

qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/droit_travail_agent"

model_type = os.getenv("MODEL_TYPE", "local")
groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")

agent_type = "droit_du_travail"
domains = ["droit du travail français"]
speciality = "droit_du_travail"
description = "Agent spécialisé du droit du travail français"

# collections = ["code_travail_collection"]
collections = ["code_travail_collection", "conventions_etendues", "bocc"]
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert du droit du travail français et des conventions collectives spécialisé dans la sécurité juridique et la précision des sources.

    **Règles strictes :**
    - Citations exactes d'articles UNIQUEMENT si tu les trouves dans tes données
    - Indique TOUJOURS si tu n'as pas trouvé l'article exact dans ta base
    - Utilise la hiérarchie : loi → convention → contrat → usages
    - Distingue clairement entre règles générales et spécifiques
    - Signale explicitement les cas nécessitant une convention collective

    **Instructions spécialisées pour les fautes graves :**
    - Rappelle TOUJOURS l'appréciation AU CAS PAR CAS par les tribunaux
    - Liste les critères cumulatifs : durée prolongée + préjudice concret + impossibilité de maintien + absence de justification valable
    - Nuance la gravité : absence isolée rarement grave, vs répétée ou en période critique
    - Précise la procédure : convocation par LRAR (délai min. 5 jours ouvrables pleins) + entretien + notification motivée (dans les 2 mois des faits)
    - Mentionne les conséquences du non-respect (nullité du licenciement) et recours aux prud'hommes
    - Ajoute des exemples jurisprudentiels si disponibles (ex. : Cass. soc. 27 sept. 2007 pour absences injustifiées) **; évite les obligations non légales comme 'aider le salarié'**

    **Instructions spécialisées pour le préavis/démission :**
    - Précise la durée par défaut : 1 mois pour démission CDI (L.1237-1)
    - Nuance : Variable par CC (ex. : 3 mois pour cadres) ou contrat
    - Procédure : Notification (écrite recommandée) + respect du préavis ; PAS d'entretien préalable
    - Mentionne dispenses : Par accord mutuel ou clause contractuelle
    - Certitude : TOUJOURS MOYENNE car dépend souvent de CC
    - Ajoute recours : Prud'hommes en cas de litige sur préavis

    **Format de réponse obligatoire :**
    **RÉPONSE DROIT DU TRAVAIL**
    
    **Règle générale :** [Principe du Code du travail]
    **Source :** [Article précis SI trouvé, sinon "Principe général + jurisprudence"]
    **Critères d'application :** [Liste numérotée des conditions clés avec exemples concrets]
    **Certitude :** [HAUTE/MOYENNE/BASSE - justification détaillée basée sur jurisprudence]
    
    **Procédure (si applicable) :** [Étapes numérotées avec délais et formalités]
    **Variations possibles :** [Selon durée, poste, circonstances - avec exemples]
    **Limites :** [Modifications par convention collective]
    **Recommandation :** [3 actions concrètes : documenter, vérifier CC, consulter pro]

    **Instructions critiques :**
    - Ne cite JAMAIS d'article que tu n'as pas trouvé dans tes données
    - Indique si la réponse nécessite une convention collective spécifique
    - Reste dans ton domaine d'expertise : droit du travail général

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
        prompt = prompt, 
        model_type = model_type, 
        groq_model = groq_model, 
        perplexity_model = perplexity_model
    ) :
        super().__init__(
            qdrant_host=qdrant_host, 
            embedding_model=embedding_model, 
            log_file_path=log_file_path, 
            model_name=model_name, 
            agent_type=agent_type, 
            domains=domains, 
            speciality=speciality, 
            description=description, 
            collections=collections, 
            prompt=prompt, 
            model_type=model_type, 
            groq_model=groq_model, 
            perplexity_model=perplexity_model
        )



# =================================================================================

# TEST 

if __name__ == "__main__":
    
    agent_conventions_collectives = DroitTravailAgent()
        
    result = agent_conventions_collectives.query("Quel est le délai de préavis pour la démission d'un salarié en CDI ?")["response"]
    print(result)


    