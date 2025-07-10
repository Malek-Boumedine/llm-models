import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_and_read_excel_files, load_pdf_files, chunk_and_insert_pdf_file, split_texts, add_chunks_to_db, log_and_print

from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client, create_collection, disable_indexing, reactivate_indexing
from datetime import datetime




# ==============================================================================================================

os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"
load_dotenv()


client_host = os.getenv("QDRANT_HOST", "http://localhost:6333")

# embedding_model = "mxbai-embed-large"
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)

files_path = os.path.join("../1.scraping_data/data/code_du_travail/")

# ================================================================================================================

# ingestion des icdd avec correspondance ape et du code du travail

def ingest_idcc_code_travail(pdf_path : str = files_path) -> int:
    
    code_travail_col_name = "code_travail_collection"
    idcc_ape_col_name = "idcc_ape_collection"
    code_travail_separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
    idcc_separators = ["\n"]
    
    logs_dir = "logs/idcc_code_travail/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_name = f"{logs_dir}log_{now_str}.log"

    with open(logfile_name, "w", encoding="utf-8") as logfile:

        pdf_documents = load_pdf_files(pdf_path)
        if pdf_documents : 
            log_and_print(f"Fichier(s) pdf chargés avec succès ! {len(pdf_documents)} fichiers chargés \n", logfile = logfile)
        
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
        embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)
        
        client = get_qdrant_client(client_host, logfile = logfile)
        create_collection(client = client, collection_name = idcc_ape_col_name, embedding_function = embedding_function, logfile = logfile)
        disable_indexing(client = client, collection_name = idcc_ape_col_name, logfile = logfile)
        
        create_collection(client = client, collection_name = code_travail_col_name, embedding_function = embedding_function, logfile = logfile)
        disable_indexing(client = client, collection_name = code_travail_col_name, logfile = logfile)
        
        try:
            if not pdf_documents: 
                log_and_print(f"Aucun PDF trouvé dans {pdf_path}", logfile)
                return 0
            else:
                for i, file in enumerate(pdf_documents, 1): 
                    file_name = os.path.splitext(os.path.basename(file))[0] 
                    log_and_print("="*50+"\n", logfile )
                    log_and_print(f"Fichier {i}/{len(pdf_documents)}", logfile )
                    log_and_print(f"Fichier : {file_name} \n", logfile )
                    if file_name == "code_du_travail":
                        chunk_and_insert_pdf_file(
                            client=client,
                            collection=code_travail_col_name,
                            embedding_function=embedding_function,
                            file_path=file, 
                            separators=code_travail_separators)
                        log_and_print(f"{file_name} ajouté avec succès à la collection {code_travail_col_name}", logfile)
                    elif file_name == "IDCC_liste":
                        chunk_and_insert_pdf_file(
                            client=client,
                            collection=idcc_ape_col_name,
                            embedding_function=embedding_function,
                            file_path=file, 
                            separators=idcc_separators)
                        log_and_print(f"{file_name} ajouté avec succès à la collection {idcc_ape_col_name}", logfile)
                reactivate_indexing(client = client, collection_name = idcc_ape_col_name, logfile = logfile)
                reactivate_indexing(client = client, collection_name = code_travail_col_name, logfile = logfile)
            return 1
        
        except Exception as e:
            log_and_print(f"Erreur de traitement : {e}", logfile)
            log_and_print("\n"+"="*50+"\n", logfile)
            return 0

# ================================================================================================================

def ingest_idcc_ape_excel(files_path: str = files_path, client_host : str = client_host) -> int:
    
    idcc_ape_col_name = "idcc_ape_collection"
    ape_idcc_separators = ["\n"]
    
    logs_dir = "logs/idcc_code_travail/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_name = f"{logs_dir}log_{now_str}.log"

    with open(logfile_name, "w", encoding="utf-8") as logfile:

        excel_documents = load_and_read_excel_files(files_path)
        if excel_documents : 
            log_and_print(f"Fichier(s) excel chargés avec succès ! {len(excel_documents)} fichiers chargés \n", logfile)

        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
        embedding_function = ef.OllamaEmbeddingFunction(model_name=embedding_model)
        
        client = get_qdrant_client(client_host, logfile = logfile)
        create_collection(client=client, collection_name = idcc_ape_col_name, embedding_function=embedding_function, logfile = logfile)
        disable_indexing(client = client, collection_name = idcc_ape_col_name, logfile = logfile)

        try:
            # Filtrer les documents Excel
            idcc_excel_doc = [d for d in excel_documents if d["file"] == "IDCC_APE.xls"]
            
            if not idcc_excel_doc: 
                log_and_print("Aucun fichier Excel trouvé dans", files_path, logfile = logfile)
                return 0
            else:
                file_name = idcc_excel_doc[0]["file"]
                file_text = idcc_excel_doc[0]["text"]
                log_and_print("="*50+"\n", logfile = logfile)
                log_and_print(f"Fichier : {file_name} \n", logfile = logfile)
                
                # Créer les chunks
                ape_idcc_chunks = split_texts(
                    text=file_text,
                    separators=ape_idcc_separators,
                    chunk_size=2000, 
                    chunk_overlap=200)
                
                success = add_chunks_to_db(
                    client=client,                    
                    collection=idcc_ape_col_name,     
                    chunks=ape_idcc_chunks,           
                    file_name=file_name,              
                    embedding_function=embedding_function)
                
                if success:
                    log_and_print(f"Insertion réussie pour {file_name} ({len(ape_idcc_chunks)} chunks)", logfile = logfile)
                    reactivate_indexing(client = client, collection_name = idcc_ape_col_name, logfile = logfile)
                    return 1
                else:
                    log_and_print(f"Erreur lors de l'insertion pour {file_name}", logfile = logfile)
                    return 0
                    
        except Exception as e:
            log_and_print(f"Erreur de traitement : {e}", logfile = logfile)
            log_and_print("\n"+"="*50+"\n", logfile = logfile)
            return 0


    
# ================================================================================================================

if __name__ == "__main__":
    
    ingest_idcc_code_travail()
    ingest_idcc_ape_excel()
