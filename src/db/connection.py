from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient



client_host = "http://localhost:6333"

def get_qdrant_client(client_host: str) -> QdrantClient | None:
    try:
        client = QdrantClient(client_host)
        return client
    except Exception as e:
        print(f"Erreur de connexion au client {client_host} : {e}")
        print("Veuillez démarrer le service ou lancer votre container de base de données Qdrant")
        return None


client = get_qdrant_client(client_host)

# ===================================================================================================

# création d'une collection

def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 100):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return collection_name
   
   




