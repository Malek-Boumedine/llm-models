import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import Qdrant, QdrantVectorStore, RetrievalMode
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama 
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity

import warnings
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")

load_dotenv()


# ================================================================================================

model_type=os.getenv("MODEL_TYPE", "cloud")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", None)


# ================================================================================================


class BaseAgent:
    
    def __init__(self, qdrant_host, embedding_model, log_file_path, model_name, agent_type, domains, speciality, description, collections, prompt, model_type=model_type, groq_model=GROQ_MODEL, perplexity_model=PERPLEXITY_MODEL):
        self.qdrant_host = qdrant_host
        self.embedding_model = embedding_model
        self.log_file_path = log_file_path
        self.model_name = model_name
        self.groq_model = groq_model
        self.perplexity_model = perplexity_model
        
        self.agent_type = agent_type
        self.domains = domains
        self.speciality = speciality
        self.description = description
        
        self.collections = collections
        self.prompt = prompt
        self.model_type = model_type
        
        # initialisation de l'agent
        self.agent = self._initialize_agent()


    def _initialize_agent(self):
        
        # Connexion à Qdrant
        with open(self.log_file_path, "w", encoding="utf-8") as logfile:
            # Utilisez OllamaEmbeddingFunction au lieu de SentenceTransformerEmbeddings
            embedding_function = OllamaEmbeddings(model=self.embedding_model)
            client = get_qdrant_client(self.qdrant_host, logfile)
        
        # Création des retrievers
        retrievers = [
            QdrantVectorStore(
                client=client,
                collection_name=col,
                embedding=embedding_function,
                retrieval_mode=RetrievalMode.DENSE,
                content_payload_key="page_content",
                metadata_payload_key="metadata"
            ).as_retriever(search_kwargs={"k": 5})
            for col in self.collections]
        
        # Création des outils de recherche
        retriever_tools = [
            create_retriever_tool(
                ret,
                name=f"recherche_{col}",
                description=f"Recherche dans la collection {col}")
            for ret, col in zip(retrievers, self.collections)]
        
        # Modèle et mémoire
        if self.model_type == "local" :
            model = ChatOllama(model=self.model_name)
        
        elif self.model_type == "groq" :
            groq_api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
            if not groq_api_key :
                raise ValueError("GROQ_API_KEY n'est pas définie dans les variables d'environnement")
            
            model = ChatGroq(
                model=self.groq_model,
                api_key=groq_api_key,
                temperature=0.2,
                max_tokens=2200,
                top_p=1,
                stop=None)
                
        elif self.model_type == "perplexity":
            pplx_key = os.getenv("PERPLEXITY_API_KEY", PERPLEXITY_API_KEY)
            if not pplx_key :
                raise ValueError("PERPLEXITY_API_KEY n'est pas définie dans les variables d'environnement")
            
            model = ChatPerplexity(
                model=self.perplexity_model,
                pplx_api_key=pplx_key,
                temperature=0.2,
                max_tokens=2200,
                top_p=1,
                streaming=False) 
        
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}. Utilisez 'local', 'groq' ou 'perplexity'")
        
        agent_memory = MemorySaver()
        
        # Création de l'agent ReAct
        return create_react_agent(
            model,
            retriever_tools,
            prompt=self.prompt, 
            checkpointer=agent_memory)


    def query(self, question,  thread_id="th_000"):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.agent.invoke({"messages": [("user", question)]}, config)
            
            return {
                "success": True,
                "response": result["messages"][-1].content,
                "agent_type": self.agent_type,
                "thread_id": thread_id,
                "collections_used": self.collections,
                "message_count": len(result["messages"])
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e),
                "agent_type": self.agent_type,
                "thread_id": thread_id
            }


    def print_stream(self, question, thread_id="th_000"):

        result = self.query(question, thread_id)
        if result["success"]:
            print(result["response"])
        else:
            print(f"Erreur: {result['error']}")
    
    
    def get_capabilities(self):

        return {
            "speciality": self.speciality,
            "domains": self.domains,
            "collections": self.collections,
            "agent_type": self.agent_type,
            "description": self.description
        }


# =================================================================================================

