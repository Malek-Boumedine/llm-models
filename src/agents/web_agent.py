from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.agents.base_agent import BaseAgent
import os
from langchain_tavily import TavilySearch, TavilyExtract
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama 
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity
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

model_type = os.getenv("MODEL_TYPE", "local")
groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", None)

agent_type = "web_agent"
domains = ["recherche web", "actualités en temps réel", "données gouvernementales", "APIs publiques", "veille informationnelle"]
speciality = "recherche internet"
description = "Agent spécialisé dans la recherche d'informations en temps réel sur Internet et l'interrogation d'APIs gouvernementales officielles pour fournir des données actualisées et fiables"

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """Tu es un agent de recherche web spécialisé dans les sources officielles françaises et la vérification d'actualité juridique.

    **Priorités absolues :**
    - Sources officielles (.gouv.fr, Légifrance, organismes publics) UNIQUEMENT
    - Informations récentes avec dates précises
    - Vérification croisée des sources
    - Transparence sur la fiabilité des informations

    **Format de réponse obligatoire :**
    **RÉPONSE RECHERCHE WEB**
    
    **Information trouvée :** [Réponse factuelle]
    **Date :** [Date précise de l'information]
    **Source :** [URL et organisme officiel]
    **Fiabilité :** [HAUTE si .gouv.fr / MOYENNE si organisme reconnu / BASSE si autre]
    
    **Limites :**
    - Cette information web peut nécessiter validation juridique
    - Consulter les textes officiels pour application pratique

    **Instructions critiques :**
    - Utilise UNIQUEMENT des sources officielles françaises
    - Vérifie la date des informations (privilégie le récent)
    - Indique si l'information est provisoire ou définitive
    - Ne réponds que si tu trouves des sources fiables

    Réponds en français avec un ton professionnel et rigoureux.
    """),
    ("placeholder", "{messages}"),
])


class WebAgent(BaseAgent):
    
    def __init__(self, 
        qdrant_host=None, 
        embedding_model=None, 
        log_file_path=log_file_path, 
        model_name=model_name, 
        agent_type=agent_type, 
        domains=domains, 
        speciality=speciality, 
        description=description, 
        collections=None, 
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
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    def _initialize_agent(self):
        
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tools = []
        
        # Configuration du modèle selon le type
        if self.model_type == "local":
            model = ChatOllama(model=self.model_name)
        elif self.model_type == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY n'est pas définie dans les variables d'environnement")
            model = ChatGroq(
                model=self.groq_model,
                api_key=GROQ_API_KEY,
                temperature=0.2,
                max_tokens=2200,
                top_p=1,
                stop=None
            )
        elif self.model_type == "perplexity":
            if not PERPLEXITY_API_KEY:
                raise ValueError("PERPLEXITY_API_KEY n'est pas définie dans les variables d'environnement")
            model = ChatPerplexity(
                model=self.perplexity_model,
                pplx_api_key=PERPLEXITY_API_KEY,
                temperature=0.2,
                max_tokens=2200,
                streaming=False
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}. Utilisez 'local', 'groq' ou 'perplexity'")
        
        agent_memory = MemorySaver()        

        tavily_search_tool = TavilySearch(
            max_results=15, 
            topic="general",
            extract_depth="advanced",
            search_depth="advanced")
        tools.append(tavily_search_tool)

        tavily_extract_tool = TavilyExtract()
        tools.append(tavily_extract_tool)

        # # tool pour chercher des informations sur l'API
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
            prompt=self.prompt, 
            checkpointer=agent_memory)
        
        return agent
    

# =================================================================================

if __name__ == "__main__":
    
    web_agent = WebAgent()
    resultat = web_agent.query("Quelles sont les nouveautés du code du travail en 2025 ?")
    print(resultat)


