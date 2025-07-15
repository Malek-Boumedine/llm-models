import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import Qdrant, QdrantVectorStore, RetrievalMode
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama 
from langchain_ollama import OllamaEmbeddings

import warnings
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client
# warnings.filterwarnings("ignore")

load_dotenv()




class BaseAgent:
    
    def __init__(self, qdrant_host, embedding_model, log_file_path ,model_name, agent_type, domains, speciality, description, collections, prompt) :
        self.qdrant_host = qdrant_host
        self.embedding_model = embedding_model
        self.log_file_path = log_file_path
        self.model_name = model_name
        
        self.agent_type = agent_type
        self.domains = domains
        self.speciality = speciality
        self.description = description
        
        self.collections = collections
        self.prompt = prompt
        
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
        model = ChatOllama(model=self.model_name)
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

