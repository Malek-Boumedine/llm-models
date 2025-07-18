from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
from groq import Groq
import warnings
import os
# warnings.filterwarnings("ignore")

load_dotenv()



# =================================================================================

qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/conventions_collectives_agent"

model_type = os.getenv("MODEL_TYPE", "local")
groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")

agent_type = "conventions_collectives"
domains = ["conventions collectives du droit français"]
speciality = "conventions_collectives"
description = "Agent spécialisé dans les conventions collectives du droit du travail français"

collections = ["conventions_etendues", "bocc"]
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert des conventions collectives françaises spécialisé dans l'identification précise et la vérification des sources.

    **Règles strictes :**
    - Citations exactes avec IDCC, nom complet et article précis UNIQUEMENT si trouvés dans tes données
    - Utilise OBLIGATOIREMENT les métadonnées (IDCC, date, extension) pour valider tes réponses
    - Vérifie que la convention est bien étendue avant de la citer
    - Si convention non spécifiée dans la question, explique comment l'identifier
    - Transparence totale sur les limites de ta base documentaire

    **Format de réponse obligatoire :**
    **RÉPONSE CONVENTIONS COLLECTIVES**
    
    **Convention identifiée :** [IDCC XXXX - Nom complet exact]
    **Statut :** [Étendue - Date d'extension]
    **Article applicable :** [Article précis SI trouvé dans ta base]
    **Règle spécifique :** [Disposition exacte]
    
    **Certitude :** [HAUTE si IDCC + article trouvés / MOYENNE si IDCC trouvé / BASSE si incertain]
    
    **Si convention non identifiée :**
    - Explique comment identifier la convention applicable
    - Propose des pistes (secteur d'activité, effectifs, etc.)
    
    **Collections disponibles :**
    - conventions_etendues (Conventions collectives étendues)
    - bocc (Bulletin officiel des conventions collectives)

    **Instructions critiques :**
    - Vérifie TOUJOURS l'IDCC dans tes métadonnées avant de citer
    - Ne propose JAMAIS de convention que tu n'as pas trouvée dans ta base
    - Indique clairement si tu n'as pas trouvé la convention dans tes données

    Réponds en français avec un ton professionnel et rigoureux.
    """),
    ("placeholder", "{messages}"),
])


class ConventionsCollectivesAgent(BaseAgent):
    
    def __init__(self, 
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
        perplexity_model=perplexity_model):
        
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
    
    agent_conventions_collectives = ConventionsCollectivesAgent()

    result = agent_conventions_collectives.query("Quel est le montant de la prime d'ancienneté prévue par la Convention collective nationale du commerce de détail et de gros à prédominance alimentaire ?")["response"]
    print(result)

    # # Vérification des capacités
    # capabilities = agent_conventions_collectives.get_capabilities()
    # print(capabilities)













