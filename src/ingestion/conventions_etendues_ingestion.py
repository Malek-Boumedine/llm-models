import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_pdf_files, chunk_and_insert_pdf_file

from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from db.connection import get_qdrant_client, create_collection, disable_indexing, reactivate_indexing



# ==============================================================================================================

os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"
load_dotenv()


client_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
client = get_qdrant_client(client_host)

# embedding_model = "mxbai-embed-large"
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)

files_path = os.path.join("../1.scraping_data/data/conventions_etendues/")

# ==============================================================================================================

# ingestion de toutes les conventions étendues

def ingestion_conventions_etendues(pdf_path: str = files_path, embedding_model: str = embedding_model) -> int:
    
    collection_name = "conventions_etendues"
    separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
    
    pdf_documents = load_pdf_files(pdf_path)
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
    embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)

    create_collection(client = client, collection_name = collection_name, embedding_function = embedding_function)
    disable_indexing(client = client, collection_name = collection_name)

    try:
        if not pdf_documents:
            print("Aucun PDF trouvé dans", pdf_path)
        else:
            print(f"{len(pdf_documents)} fichiers trouvés dans le répertoire {pdf_path}\n")
            for i, file in enumerate(pdf_documents, 1):
                file_name = os.path.splitext(os.path.basename(file))[0]
                print("="*50, "\n")
                print(f"Fichier {i}/{len(pdf_documents)}")
                print(f"Fichier : {file_name} \n")
                chunk_and_insert_pdf_file(
                    client=client,
                    collection=collection_name,
                    embedding_function=embedding_function,
                    file_path=file, 
                    separators=separators)
                print(f"{file_name} ajouté avec succès à la collection {collection_name}")
            reactivate_indexing(client = client, collection_name = collection_name)
        return 1

    except Exception as e:
        print(f"Erreur de traitement : {e}")
        print("\n", "="*50, "\n")
        return 0

# ==============================================================================================================

if __name__ == "__main__":
    ingestion_conventions_etendues()

