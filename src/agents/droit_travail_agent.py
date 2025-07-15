import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama 

from langchain_community.vectorstores import Chroma
import warnings
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")

load_dotenv()



# =================================================================================

# # Connexion à Qdrant (exemple en local)
# with open("fichier_log_agent_test", "w", encoding="utf-8") as logfile:
#     qdrant_host = "http://localhost:6333"
#     embedding_model = "paraphrase-multilingual-mpnet-base-v2"
#     embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
#     client = get_qdrant_client(qdrant_host, logfile)

# collections = ["code_travail_collection", "conventions_etendues", "bocc", "idcc_ape_collection"]
# retriever = [
#     Qdrant(
#         client=client,
#         collection_name=col,
#         embeddings=embedding_function
#     ).as_retriever(search_kwargs={"k": 5})
#     for col in collections]

# # =================================================================================

# # Création de l'outil LangChain pour la recherche
# retriever_tools = [
#     create_retriever_tool(
#         ret,
#         name=f"recherche_{col}",
#         description=f"Recherche dans la collection {col}")
#     for ret, col in zip(retriever, collections)]

# tools = retriever_tools

# # Prompt contextuel
# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      """Tu es un expert en droit du travail français.
#      Tu réponds toujours en t'appuyant exclusivement sur les informations contenues dans ta base documentaire.
     
#      Tes collections disponibles sont :
#      - code_travail_collection
#      - conventions_etendues
#      - bocc
#      - idcc_ape_collection

#      Utilise la méthode ReAct suivante :
#      Question: la question initiale de l'utilisateur
#      Thought: explique ta réflexion sur comment trouver la réponse
#      Action: utilise l'outil correspondant à la collection la plus pertinente
#      Observation: les résultats de l'outil
#      Thought: ta réflexion suite aux résultats
#      Final Answer: réponds à l'utilisateur de manière claire, détaillée et structurée en français.

#      Si aucune information n'est trouvée, indique-le clairement.
#      """),
#     ("placeholder", "{messages}"),
#     ("user", "{input}"),
# ])

# # model = ChatOllama(model="mistral:7b")
# model = ChatOllama(model="llama3.1:latest")
# agent_memory = MemorySaver()

# agent = create_react_agent(
#     model,
#     tools,
#     checkpointer=agent_memory)


# def print_stream(agent, question):
#     config = {"configurable": {"thread_id": "abc123"}}
#     result = agent.invoke({"messages": [("user", question)]}, config)
#     print(result["messages"][-1].content)

# # ==============================================================================================================

# if __name__ == "__main__":
    
#     while True :
#         print("\n" + "="*60 + "\n")
#         question = input("User : ")
#         if question == "q" : 
#             break
#         print("\n" + "Agent : " + "\n")
#         print_stream(
#             agent,
#             f"{question}"
#         )
    
# # ==============================================================================================================
# # ==============================================================================================================

# agent droit du travail

qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
log_file_path = "logs/droit_travail_agent"

class DroitTravailAgent:
    """Agent spécialisé en droit du travail français avec pattern Factory"""
    
    def __init__(self, qdrant_host=qdrant_host, embedding_model=embedding_model, log_file_path="log_file_path" ,model_name=model_name) :
        """
        Initialise l'agent de droit du travail
        
        Args:
            qdrant_host: URL du serveur Qdrant
            embedding_model: Modèle d'embedding à utiliser
            log_file_path: Chemin du fichier de log
            model_name: Nom du modèle Ollama à utiliser
        """
        self.qdrant_host = qdrant_host
        self.embedding_model = embedding_model
        self.log_file_path = log_file_path
        self.model_name = model_name
        
        # Collections disponibles
        # self.collections = ["code_travail_collection", "conventions_etendues", "bocc", "idcc_ape_collection"]
        self.collections = ["code_travail_collection", "bocc"]
        
        # Initialisation de l'agent
        self.agent = self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialise tous les composants de l'agent"""
        
        # Connexion à Qdrant
        with open(self.log_file_path, "w", encoding="utf-8") as logfile:
            embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model)
            client = get_qdrant_client(self.qdrant_host, logfile)
        
        # Création des retrievers
        retrievers = [
            Qdrant(
                client=client,
                collection_name=col,
                embeddings=embedding_function
            ).as_retriever(search_kwargs={"k": 8})
            for col in self.collections]
        
        # Création des outils de recherche
        retriever_tools = [
            create_retriever_tool(
                ret,
                name=f"recherche_{col}",
                description=f"Recherche dans la collection {col}")
            for ret, col in zip(retrievers, self.collections)
        ]
        
        # Prompt contextuel
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """Tu es un expert en droit du travail français.
            Tu réponds toujours en t'appuyant exclusivement sur les informations contenues dans ta base documentaire.
             
            Tes collections disponibles sont :
            - code_travail_collection
            - bocc
             
            IMPORTANT : Utilise les métadonnées des documents pour :
            - Citer les sources précises (article, numéro, date)
            - Identifier le type de document (loi, décret, arrêté)
            - Mentionner la date de publication si pertinente

            Utilise la méthode ReAct suivante :
            Question: la question initiale de l'utilisateur
            Thought: explique ta réflexion sur comment trouver la réponse
            Action: utilise l'outil correspondant à la collection la plus pertinente
            Observation: les résultats de l'outil
            Thought: ta réflexion suite aux résultats
            Final Answer: réponds à l'utilisateur de manière claire, détaillée et structurée en français.

            Si aucune information n'est trouvée, indique-le clairement.
            """),
            ("placeholder", "{messages}"),
            ("user", "{input}"),
        ])
        
        # Modèle et mémoire
        model = ChatOllama(model=self.model_name)
        agent_memory = MemorySaver()
        
        # Création de l'agent ReAct
        return create_react_agent(
            model,
            retriever_tools,
            prompt=prompt,
            checkpointer=agent_memory)
    
    def query(self, question, thread_id="th_000"):
        """
        Interface standardisée pour interroger l'agent
        
        Args:
            question: Question à poser à l'agent
            thread_id: Identifiant de thread pour la mémoire
            
        Returns:
            dict: Réponse structurée avec métadonnées
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.agent.invoke({"messages": [("user", question)]}, config)
            
            return {
                "success": True,
                "response": result["messages"][-1].content,
                "agent_type": "droit_travail",
                "thread_id": thread_id,
                "collections_used": self.collections,
                "message_count": len(result["messages"])
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "agent_type": "droit_travail",
                "thread_id": thread_id
            }
    
    def print_stream(self, question, thread_id="default_thread"):
        """
        Affiche directement la réponse (compatible avec votre usage actuel)
        
        Args:
            question: Question à poser
            thread_id: Identifiant de thread
        """
        result = self.query(question, thread_id)
        if result["success"]:
            print(result["response"])
        else:
            print(f"Erreur: {result['error']}")
    
    def get_capabilities(self):
        """
        Retourne les capacités de l'agent pour l'agent maître
        
        Returns:
            dict: Description des capacités
        """
        return {
            "agent_type": "droit_travail",
            "domains": ["droit du travail français"],
            "collections": self.collections,
            "model": self.model_name,
            "description": "Expert en droit du travail français avec accès aux collections juridiques"
        }
    
    def health_check(self):
        """
        Vérification de l'état de l'agent
        
        Returns:
            dict: État de santé de l'agent
        """
        try:
            # Test simple pour vérifier que l'agent fonctionne
            test_result = self.query("Test de fonctionnement", "health_check")
            return {
                "status": "healthy" if test_result["success"] else "unhealthy",
                "agent_type": "droit_travail",
                "collections_count": len(self.collections),
                "last_check": "now"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent_type": "droit_travail",
                "error": str(e)
            }


# Factory function pour créer l'agent (optionnel)
def create_droit_travail_agent(**kwargs):
    """
    Factory function pour créer une instance de l'agent
    
    Args:
        **kwargs: Arguments de configuration
        
    Returns:
        DroitTravailAgent: Instance de l'agent
    """
    return DroitTravailAgent(**kwargs)





    