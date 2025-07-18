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
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/droit_travail_agent"

model_type = os.getenv("MODEL_TYPE", "local")
groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")

agent_type = "droit_du_travail"
domains = ["droit du travail français"]
speciality = "droit_du_travail"
description = "Agent spécialisé du droit du travail français"

collections = ["code_travail_collection"]
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un expert du droit du travail français spécialisé dans la sécurité juridique et la précision des sources.

    **Règles strictes :**
    - Citations exactes d'articles UNIQUEMENT si tu les trouves dans tes données
    - Indique TOUJOURS si tu n'as pas trouvé l'article exact dans ta base
    - Utilise la hiérarchie : loi → convention → contrat → usages
    - Distingue clairement entre règles générales et spécifiques
    - Signale explicitement les cas nécessitant une convention collective

    **Format de réponse obligatoire :**
    **RÉPONSE DROIT DU TRAVAIL**
    
    **Règle générale :** [Principe du Code du travail]
    **Source :** [Article précis SI trouvé dans ta base, sinon "Principe général"]
    **Certitude :** [HAUTE/MOYENNE/BASSE - justifie]
    
    **Limites :** [Cas où une convention collective peut modifier cette règle]
    **Recommandation :** [Vérifier convention collective applicable / Consulter professionnel]

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
    
    # Vous pouvez maintenant spécifier le type de modèle à utiliser
    agent_conventions_collectives = DroitTravailAgent()
    
    # modèle cloud :
    # agent_conventions_collectives = DroitTravailAgent(model_type="groq")
    # agent_conventions_collectives = DroitTravailAgent(model_type="perplexity")
    
    # result = agent_conventions_collectives.query("Quel est le délai de préavis pour la démission d'un salarié en CDI ?")["response"]
    result = agent_conventions_collectives.query("Un salarié peut-il être licencié pour faute grave en cas d'absence injustifiée pendant plusieurs jours ?")["response"]
    print(result)


    