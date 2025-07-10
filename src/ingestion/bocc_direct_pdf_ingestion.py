import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_pdf_files, chunk_and_insert_pdf_file, log_and_print

from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client, create_collection, disable_indexing, reactivate_indexing
from datetime import datetime




# ==============================================================================================================

os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"
load_dotenv()

client_host = os.getenv("QDRANT_HOST", "http://localhost:6333")

files_path = os.path.join("../1.scraping_data/data/BOCC_pdf_direct_link/")

# ==============================================================================================================

# ingestion de tous les bulletins officiels des conventions étendues (avec lien pdf direct)

def ingest_direct_pdf_bocc(pdf_path: str = files_path, client_host : str = client_host) -> int:
    
    collection_name = "bocc"
    separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
    
    logs_dir = "logs/bocc_logs/direct_pdf/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_name = f"{logs_dir}log_{now_str}.log"
    
    with open(logfile_name, "w", encoding="utf-8") as logfile:

        pdf_documents = load_pdf_files(pdf_path)
        if pdf_documents : 
            log_and_print(f"Fichiers pdf chargés avec succès ! {len(pdf_documents)} fichiers chargés \n", logfile)
        
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
        embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)
        
        client = get_qdrant_client(client_host, logfile = logfile)
        create_collection(client = client, collection_name = collection_name, embedding_function = embedding_function, logfile = logfile)
        disable_indexing(client = client, collection_name = collection_name, logfile = logfile)

        try:
            if not pdf_documents:
                log_and_print(f"Aucun PDF trouvé dans {pdf_path}" , logfile)
                return 0
            else:
                for i, file in enumerate(pdf_documents, 1):
                    file_name = os.path.splitext(os.path.basename(file))[0]
                    log_and_print("="*80+"\n", logfile)
                    log_and_print(f"Fichier {i}/{len(pdf_documents)}", logfile)
                    log_and_print(f"Fichier : {file_name} \n", logfile)
                    chunk_and_insert_pdf_file(
                        client=client,
                        collection=collection_name,
                        embedding_function=embedding_function,
                        file_path=file, 
                        separators=separators)
                    log_and_print(f"{file_name} ajouté avec succès à la collection {collection_name}", logfile)
                reactivate_indexing(client = client, collection_name = collection_name, logfile = logfile)
            return 1

        except Exception as e:
            log_and_print(f"Erreur de traitement : {e}", logfile)
            log_and_print("\n"+"="*80+"\n", logfile)
            return 0

# ==============================================================================================================

if __name__ == "__main__" : 
    
    ingest_direct_pdf_bocc()

