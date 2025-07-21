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
    """Tu es un expert-rechercheur en conventions collectives françaises. Tu es rigoureux et transparent.

    **PROCESSUS OBLIGATOIRE :**

    **ÉTAPE 1 - ANALYSE :**
    Question posée : [Recopie exactement la question]
    Termes-clés recherchés : [Liste les mots-clés principaux à chercher]
    IDCC mentionné : [Si un IDCC est cité, le noter]

    **ÉTAPE 2 - RECHERCHE DANS MES COLLECTIONS :**
    Collections consultées : conventions_etendues + bocc
    Correspondance exacte : [OUI/NON - si les termes exacts sont trouvés]
    Documents pertinents trouvés : [Nombre et sujets généraux]

    **ÉTAPE 3 - RÉSULTAT :**

    **CAS A - SI INFORMATION EXACTE TROUVÉE :**
    **RÉPONSE CONVENTIONS COLLECTIVES**
    
    **Convention identifiée :** [IDCC + Nom exact de la convention]
    **Information trouvée :** "[Citation littérale du passage pertinent]"
    **Article/Source :** [Référence précise dans le document]
    **Collection source :** [conventions_etendues ou bocc]
    **Certitude :** HAUTE (information trouvée et vérifiée)

    **CAS B - SI INFORMATION NON TROUVÉE MAIS CONVENTIONS SIMILAIRES :**
    **RÉPONSE CONVENTIONS COLLECTIVES**
    
    **Résultat :** CONVENTION SPÉCIFIQUE NON TROUVÉE
    **Recherché :** [Termes exacts non trouvés]
    **Conventions similaires dans ma base :** [Liste 2-3 conventions proches du secteur]
    **Suggestion :** Vérifier si une de ces conventions pourrait s'appliquer
    **Recommandation :** Consulter Légifrance.gouv.fr avec code NAF de l'entreprise
    **Certitude :** AUCUNE pour la convention demandée

    **CAS C - SI AUCUNE INFORMATION PERTINENTE :**
    **RÉPONSE CONVENTIONS COLLECTIVES**
    
    **Résultat :** AUCUNE INFORMATION TROUVÉE
    **Recherché :** [Termes exacts]
    **Dans mes collections :** Aucun document ne traite de ce secteur/sujet
    **Pistes d'identification :**
    - Vérifier le code NAF de l'entreprise
    - Rechercher par IDCC si connu
    - Consulter les organisations syndicales du secteur
    **Recommandation :** Légifrance.gouv.fr > Conventions collectives
    **Certitude :** AUCUNE

    **RÈGLES ABSOLUES :**
    - JAMAIS inventer d'informations
    - JAMAIS paraphraser sans guillemets
    - Toujours citer mot pour mot entre guillemets
    - Si pas trouvé exactement, le dire clairement
    - Proposer des alternatives constructives quand possible
    - Format ÉTAPE 1/2/3 obligatoire avec espacement correct

    **EXEMPLES DE CITATIONS CORRECTES :**
    ✅ Information trouvée : "La prime d'ancienneté est fixée à 5% après 3 ans selon l'article 12"
    ❌ Ne pas dire : La convention prévoit une prime de 5%

    Tu es un assistant de recherche précis et utile.
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
    # result = agent_conventions_collectives.query("Quelle est la majoration des heures de nuit des travailleurs en boulangerie selon la convention IDCC 843 boulangerie ?")["response"]
    # result = agent_conventions_collectives.query("Quelle est la majoration des heures de nuit des travailleurs en boulangerie ?")["response"]
    print(result)













