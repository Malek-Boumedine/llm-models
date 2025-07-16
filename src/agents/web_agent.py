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
log_file_path = "logs/agents/web_agent"

agent_type = "web_agent"
domains = ["recherche web", "actualités en temps réel", "données gouvernementales", "APIs publiques", "veille informationnelle"]
speciality = "recherche internet "
description = "Agent spécialisé dans la recherche d'informations en temps réel sur Internet et l'interrogation d'APIs gouvernementales officielles pour fournir des données actualisées et fiables"

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un agent de recherche web spécialisé dans les sources officielles françaises.

    **Priorités :**
    - Sources officielles (.gouv.fr, organismes publics)
    - Informations récentes avec dates
    - Citations précises des sources

    **Format :**
    - Réponse directe et factuelle
    - Sources avec URLs et dates
    - Niveau de fiabilité indiqué

    Réponds en français avec un ton professionnel.
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
        os.environ["TAVILY_API_KEY"] = tavily_api_key


    def _initialize_agent(self):
        
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tools = []
        model = ChatOllama(model=self.model_name)
        agent_memory = MemorySaver()        

        tavily_search_tool = TavilySearch(
            max_results = 15, 
            topic="general",
            extract_depth="advanced",
            search_depth = "advanced", )
        tools.append(tavily_search_tool)

        tavily_extract_tool = TavilyExtract()
        tools.append(tavily_extract_tool)

        # # tool pour chercger des informations sur l'API
        # def recherche_entreprise_api(nom_entreprise_siren_siret):
        #     try:
        #         api_url = f"https://recherche-entreprises.api.gouv.fr/search?q={nom_entreprise_siren_siret}"
        #         response = requests.get(api_url, timeout=10)
                
        #         if response.status_code == 200:
        #             data = response.json()
        #             if data.get("total_results", 0) > 0:
        #                 result = data["results"][0]
        #                 siege = result.get("siege", {})
                        
        #                 return {
        #                     "status": "success",
        #                     "siren": result.get("siren"),
        #                     "nom_complet": result.get("nom_complet"),
        #                     "activite_principale": result.get("activite_principale"),
        #                     "adresse": siege.get("adresse"),
        #                     "code_postal": siege.get("code_postal"),
        #                     "libelle_commune": siege.get("libelle_commune"),
        #                     "date_creation": siege.get("date_creation"),
        #                     "date_debut_activite": siege.get("date_debut_activite")}
        #             else:
        #                 return {"status": "no_results", "message": "Aucun résultat trouvé"}
                
        #         else:
        #             return {"status": "error", "message": f"Erreur API: {response.status_code}"}
                    
        #     except requests.RequestException as e:
        #         return {"status": "error", "message": f"Erreur réseau: {str(e)}"}
        #     except Exception as e:
        #         return {"status": "error", "message": f"Erreur inattendue: {str(e)}"}     
                         
        # entreprise_tool = Tool(
        #     name="recherche_entreprise",
        #     description="Recherche les informations d'une entreprise via l'API gouvernementale",
        #     func=recherche_entreprise_api)
        # tools.append(entreprise_tool)

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







