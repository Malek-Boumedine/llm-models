import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import Qdrant, QdrantVectorStore, RetrievalMode
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama 
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity

import warnings
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client

load_dotenv()

# ================================================================================================

model_type=os.getenv("MODEL_TYPE", "cloud")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", None)

cloud_ollama_embedding_model = os.getenv("CLOUD_OLLAMA_EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
cloud_fallback_model = os.getenv("CLOUD_FALLBACK_MODEL", "BAAI/bge-m3")

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
            if self.model_type == "local" :
                embedding_function = OllamaEmbeddings(
                    base_url=ollama_host,
                    model=self.embedding_model
                )
            else : 
                embedding_function = HuggingFaceEmbeddings(model_name = cloud_ollama_embedding_model)
            client = get_qdrant_client(self.qdrant_host, logfile)
        
        # Création des retrievers
        retrievers = [
            QdrantVectorStore(
                client=client,
                collection_name=col,
                embedding=embedding_function,
                retrieval_mode=RetrievalMode.DENSE,
                content_payload_key="text",
                metadata_payload_key=None
            ).as_retriever(
                search_kwargs={
                    "k": 6,
                    "filter": None})
            for col in self.collections]
        
        # Stockage des retrievers pour Perplexity
        self.retrievers = retrievers
        
        # Création des outils de recherche
        retriever_tools = [
            create_retriever_tool(
                ret,
                name=f"recherche_{col}",
                description=f"Recherche dans la collection {col}")
            for ret, col in zip(retrievers, self.collections)]
        
        # Modèle et mémoire
        if self.model_type == "local":
            model = ChatOllama(
                model=self.model_name,
                temperature=0.1, 
                seed=42           )
        
        elif self.model_type == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY n'est pas définie dans les variables d'environnement")
            
            model = ChatGroq(
                model=self.groq_model,
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=2200,
                top_p=1,
                stop=None)
                
        elif self.model_type == "perplexity":
            pplx_key = os.getenv("PERPLEXITY_API_KEY", PERPLEXITY_API_KEY)
            if not pplx_key:
                raise ValueError("PERPLEXITY_API_KEY n'est pas définie dans les variables d'environnement")
            
            model = ChatPerplexity(
                model=self.perplexity_model,
                pplx_api_key=pplx_key,
                temperature=0.1,
                max_tokens=2200,
                top_p=1,
                streaming=False) 
        
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}. Utilisez 'local', 'groq' ou 'perplexity'")
        
        # Stockage du modèle
        self.llm = model
        
        # Création de l'agent ReAct (sauf pour Perplexity)
        if self.model_type == "perplexity":
            print(f"🔶 {self.agent_type}: Mode Perplexity - RAG direct")
            return None
        else:
            agent_memory = MemorySaver()
            return create_react_agent(
                model,
                retriever_tools,
                prompt=self.prompt, 
                checkpointer=agent_memory)

    def query(self, question, thread_id="th_000"):
        try:
            if self.model_type == "perplexity":
                # Mode Perplexity : RAG direct
                context_docs = []
                for retriever, col_name in zip(self.retrievers, self.collections):
                    docs = retriever.get_relevant_documents(question)
                    for doc in docs[:3]:
                        context_docs.append(f"[{col_name}] {doc.page_content}")
                
                context = "\n\n".join(context_docs[:10]) if context_docs else "Aucun contexte trouvé"
                
                rag_prompt = f"""{self.prompt}

CONTEXTE JURIDIQUE :
{context}

QUESTION : {question}

Réponds en utilisant le contexte fourni."""
                
                response = self.llm.invoke(rag_prompt)
                
                return {
                    "success": True,
                    "response": response.content,
                    "agent_type": self.agent_type,
                    "thread_id": thread_id,
                    "collections_used": self.collections,
                    "message_count": 2
                }
            else:
                # Mode normal avec agent
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
