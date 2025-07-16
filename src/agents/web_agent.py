from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
import os
from langchain_tavily import TavilySearch, TavilyExtract
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama 
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import Tool
import requests

import warnings
# warnings.filterwarnings("ignore")

load_dotenv()



# =================================================================================

tavily_api_key = os.getenv("TAVILY_API_KEY", None)

model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/web_agent"

agent_type = "web_agent"
domains = ["recherche web", "actualités en temps réel", "données gouvernementales", "APIs publiques", "veille informationnelle"]
speciality = "recherche internet "
description = "Agent spécialisé dans la recherche d'informations en temps réel sur Internet et l'interrogation d'APIs gouvernementales officielles pour fournir des données actualisées et fiables"

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un agent de recherche web spécialisé dans la collecte d'informations officielles et gouvernementales en temps réel.

    **Tes capacités principales :**
        - Recherche d'informations actualisées sur Internet via des sources fiables et officielles
        - Interrogation d'APIs gouvernementales françaises
        - Veille sur les actualités, la réglementation et les bases de données publiques
        - Synthèse claire à partir de plusieurs sources

    **Règles strictes à suivre :**
        - Privilégie TOUJOURS les sources officielles (.gouv.fr, .fr, organismes publics, APIs d'État)
        - Vérifie la fraîcheur des informations : indique systématiquement la date de consultation ou de dernière mise à jour
        - Croise plusieurs sources (3 à 5 maximum) pour confirmer les informations, et signale toute divergence
        - Cite précisément tes sources (URL complète + nom du site + date d'accès)
        - Si une information provient d'une API, mentionne l’API utilisée et la date/heure de la requête
        - Distingue clairement les faits avérés des informations à confirmer ou en cours d’actualisation
        - Si l’information recherchée n’est pas disponible ou fiable, indique-le explicitement et propose des alternatives
        - En cas d’erreur technique (timeout, API indisponible), mentionne le problème rencontré

    **Format de réponse à respecter :**
        - Fournis une réponse directe et synthétique
        - Inclus toutes les sources principales consultées, avec :
            - Le titre du document ou de l’annonce
            - L’URL exacte
            - La date de consultation ou publication
            - Un résumé de chaque source
        - Précise le niveau de fiabilité de chaque information (officielle, média généraliste, blog, etc.)
        - Si le texte intégral d’un projet de loi ou décret est disponible, donne le lien direct.
        - Présente la synthèse sous forme de liste numérotée pour chaque information trouvée.
        - Si plusieurs textes sont en discussion, précise l’état d’avancement (ex : “en projet”, “publié”, “applicable à partir du...”).
        - Termine par une recommandation claire.
    
    **Exemple de structure de réponse :**

        Synthèse :
            1. **Changement dans la procédure d'arrêt de travail** : Nouveau formulaire papier obligatoire dès le 1er septembre 2025.  
            Source : [Ameli](https://www.ameli.fr/) – Consulté le 16/07/2025
            2. **Obligations pour les employeurs face aux risques de chaleur** : Adaptation obligatoire de l’organisation du travail pour la sécurité des salariés, pauses renforcées, eau à disposition, etc.  
            Source : [Service Public](https://www.service-public.fr/) – Consulté le 16/07/2025
            3. **Financement de l'apprentissage** : Réformes à partir du 1er juillet 2025 (montants d’aide, conditions, OPCO).  
            Source : [Ministère du Travail](https://travail-emploi.gouv.fr/) – Consulté le 16/07/2025
            4. **Revalorisation des allocations chômage** : +0,5% au 1er juillet.  
            Source : [Unédic](https://www.unedic.org/) – Consulté le 16/07/2025
            5. **Législation européenne sur les congés payés** : Mise en demeure de la France, réforme attendue.  
            Source : [Commission européenne](https://ec.europa.eu/) – Consulté le 16/07/2025

        Sources principales :
            - Legifrance (https://www.legifrance.gouv.fr/) – Consulté le 16/07/2025
            - Service Public (https://www.service-public.fr/) – Consulté le 16/07/2025

        Niveau de fiabilité : élevé (sources officielles gouvernementales et institutionnelles).

        Recommandation : Vérifiez régulièrement les sites officiels pour les mises à jour et consultez un professionnel du droit du travail pour toute situation spécifique.
        
    **En cas d’absence d’information :**
        - Indique explicitement que l’information n’est pas disponible dans les sources consultées
        - Propose des sites officiels ou contacts utiles pour approfondir la recherche

        Réponds toujours en français, avec un ton professionnel et rigoureux, et tu dois fournier beaucoup de contenu, un grand nombre de caractères pour tes réponses sans synthèse.
    """),
    ("placeholder", "{messages}"),
])


class WebAgent(BaseAgent) :
    
    def __init__(self, 
        qdrant_host = None, 
        embedding_model = None, 
        log_file_path = log_file_path, 
        model_name = model_name, 
        agent_type = agent_type, 
        domains = domains, 
        speciality = speciality, 
        description = description, 
        collections = None, 
        prompt = prompt) :
        super().__init__(qdrant_host, embedding_model, log_file_path, model_name, agent_type, domains, speciality, description, collections, prompt)
        if not os.environ.get("TAVILY_API_KEY"):
            os.environ["TAVILY_API_KEY"] = tavily_api_key


    def _initialize_agent(self):
        
        tools = []
        model = ChatOllama(model=self.model_name)
        agent_memory = MemorySaver()        

        tavily_search_tool = TavilySearch(
            # include_domains=[
            #     "https://www.journal-officiel.gouv.fr/pages/accueil/", 
            #     "https://www.infogreffe.fr", 
            #     "https://www.legifrance.gouv.fr/", 
            #     "https://www.inpi.fr/fr",
            #     "https://entreprendre.service-public.fr/", 
            #     "https://bofip.impots.gouv.fr/", 
            #     "https://code.travail.gouv.fr/outils/convention-collective", 
            #     "https://code.travail.gouv.fr/outils/convention-collective/convention", 
            #     "https://code.travail.gouv.fr/outils/convention-collective/entreprise"
            # ], 
            max_results = 20, 
            topic="general",
            extract_depth="advanced",
            search_depth = "advanced", )
        tools.append(tavily_search_tool)

        tavily_extract_tool = TavilyExtract()
        tools.append(tavily_extract_tool)

        # tool pour chercger des informations sur l'API
        def recherche_entreprise_api(nom_entreprise_siren_siret):
            try:
                api_url = f"https://recherche-entreprises.api.gouv.fr/search?q={nom_entreprise_siren_siret}"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("total_results", 0) > 0:
                        result = data["results"][0]
                        siege = result.get("siege", {})
                        
                        return {
                            "status": "success",
                            "siren": result.get("siren"),
                            "nom_complet": result.get("nom_complet"),
                            "activite_principale": result.get("activite_principale"),
                            "adresse": siege.get("adresse"),
                            "code_postal": siege.get("code_postal"),
                            "libelle_commune": siege.get("libelle_commune"),
                            "date_creation": siege.get("date_creation"),
                            "date_debut_activite": siege.get("date_debut_activite")}
                    else:
                        return {"status": "no_results", "message": "Aucun résultat trouvé"}
                
                else:
                    return {"status": "error", "message": f"Erreur API: {response.status_code}"}
                    
            except requests.RequestException as e:
                return {"status": "error", "message": f"Erreur réseau: {str(e)}"}
            except Exception as e:
                return {"status": "error", "message": f"Erreur inattendue: {str(e)}"}     
                         
        entreprise_tool = Tool(
            name="recherche_entreprise",
            description="Recherche les informations d'une entreprise via l'API gouvernementale",
            func=recherche_entreprise_api)
        tools.append(entreprise_tool)

        agent = create_react_agent(
            model=model, 
            tools=tools, 
            prompt=prompt, 
            checkpointer=agent_memory)
        
        return agent
    

# =================================================================================

if __name__ == "__main__" : 
    
    web_agent = WebAgent()
    resultat = web_agent.query("Quelles sont les nouveautés du code du travail en 2025 ?")
    print(resultat)







