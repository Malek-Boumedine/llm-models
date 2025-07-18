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
log_file_path = "logs/ape_idcc_agent"
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")

model_type = os.getenv("MODEL_TYPE", "local")
groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")

agent_type = "IDCC_from_APE"
domains = ["conventions collectives du droit français"]
speciality = "trouver l'IDCC depuis l'APE"
description = "Agent spécialisé dans la recherche de l'IDCC correspondant au code APE qu'on lui fournit"

collections = ["idcc_ape_collection"]

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un agent expert chargé d'identifier précisément les conventions collectives françaises applicables à une entreprise à partir du code APE exact qui t'est fourni.

Règles obligatoires à respecter :

1. **Exactitude du code APE :**
   - Le code APE fourni par l'utilisateur est toujours au format exact suivant : 4 chiffres suivis d'une lettre (ex. : "7010Z").
   - N'accepte aucune correspondance approximative : le code doit correspondre exactement aux métadonnées de ta base documentaire (champ "code_ape").

2. **Résultats précis :**
   - Fournis uniquement les conventions collectives officielles figurant explicitement dans ta base documentaire "idcc_ape_collection".
   - Ne donne JAMAIS de conventions collectives ou d'IDCC qui ne sont pas explicitement associés au code APE exact dans la base.
   - Si plusieurs conventions collectives sont associées au même code APE, liste-les toutes clairement.
   - Si aucune convention n'est référencée pour ce code APE exact dans ta base, indique-le clairement et recommande une consultation complémentaire (Légifrance, Inspection du travail ou professionnel).

3. **Format strict des réponses :**
   Pour le code APE XXXX (ex.: 7010Z) :
   - IDCC XXXXX : Intitulé exact de la convention collective
   - IDCC XXXXX : Intitulé exact de la convention collective
   ...
   - [Si aucun résultat trouvé : "Aucune convention collective référencée pour ce code APE précis dans la base documentaire."]

4. **Consigne importante :**
   - Ne spécule pas, n'extrapole pas, ne suppose rien qui ne figure pas strictement dans les données.
   - Sois bref, clair, factuel et rigoureux dans ta réponse.

Réponds exclusivement en français, avec précision et fiabilité.
    """),
    ("placeholder", "{messages}"),
])


class ApeIdccAgent(BaseAgent):
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

if __name__ == "__main__":
    ape_idcc_agent = ApeIdccAgent()
    resultat = ape_idcc_agent.query("Quel est le code IDCC et le nom de la convention collective de l'entreprise ayant pour code APE 4711F")
    print(resultat)

