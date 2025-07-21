from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, OptimizersConfigDiff
from qdrant_client import QdrantClient
from chromadb.utils import embedding_functions
from src.utils import log_and_print



client_host = "http://localhost:6333"

# ===================================================================================================

def get_qdrant_client(client_host: str, logfile) -> QdrantClient | None:
    try:
        client = QdrantClient(client_host)
        return client
    except Exception as e:
        log_and_print(f"Erreur de connexion au client {client_host} : {e}", logfile)
        log_and_print("Veuillez démarrer le service ou lancer votre container de base de données Qdrant", logfile)
        return None

# client = get_qdrant_client(client_host)

# ===================================================================================================

def create_collection_qdrant(client: QdrantClient, collection_name: str, embedding_function, logfile : str) -> str :
    test_vector = embedding_function(["test"])[0]
    vector_size = len(test_vector)

    # with open(logfile, "w", encoding="utf-8") as logfile:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                on_disk=False,
                m=24,
                ef_construct=200,
                max_indexing_threads=6, 
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,    # Valeur normale (pas 0)
                default_segment_number=1,    # Optimisé pour mobile
                max_segment_size=3000000     # Taille réduite pour mobile
            )
        )
        log_and_print(f"Collection {collection_name} créée avec optimisations GPU (indexation active).", logfile)
    else : 
        log_and_print(f"La collection {collection_name} existe deja.", logfile)
    return collection_name

# ===================================================================================================

def disable_indexing(client : QdrantClient, collection_name : str, logfile : str) -> int :
    # with open(logfile, "w", encoding="utf-8") as logfile :
    try:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=0,        # Désactiver pendant l'insertion
                default_segment_number=1,    # Moins de segments pour mobile
                max_segment_size=3000000     # Taille réduite pour mobile
            )
        )
        log_and_print(f"Indexation désactivée pour la collection '{collection_name}'", logfile)
        return 1
    except Exception as e:
        log_and_print(f"Erreur lors de la désactivation de l'indexation pour '{collection_name}': {e}", logfile)
        return 0

# ===================================================================================================

def reactivate_indexing(client: QdrantClient, collection_name: str, logfile : str) -> int :
    # with open(logfile, "w", encoding="utf-8") as logfile :
    try :
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=20000))  # Réactiver l'indexation
        log_and_print(f"Indexation activée avec succès pour la collection '{collection_name}'", logfile)
        return 1
    except Exception as e : 
        log_and_print(f"Erreur lors de la réactivation de l'indexation pour '{collection_name}': {e}", logfile)
    
# ===================================================================================================
