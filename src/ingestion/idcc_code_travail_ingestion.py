import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_and_read_excel_files, load_pdf_files, chunk_and_insert_pdf_file, split_texts, add_chunks_to_db

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

files_path = os.path.join("../1.scraping_data/data/code_du_travail/")

# ================================================================================================================

# ingestion des icdd avec correspondance ape et du code du travail

def ingest_idcc_code_travail(pdf_path : str = files_path) -> int:
    
    code_travail_col_name = "code_travail_collection"
    idcc_ape_col_name = "idcc_ape_collection"
    code_travail_separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
    idcc_separators = ["\n"]

    pdf_documents = load_pdf_files(pdf_path)
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
    embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)
    
    create_collection(client = client, collection_name = idcc_ape_col_name, embedding_function = embedding_function)
    create_collection(client = client, collection_name = code_travail_col_name, embedding_function = embedding_function)
    
    disable_indexing(client = client, collection_name = idcc_ape_col_name)
    disable_indexing(client = client, collection_name = code_travail_col_name)
    
    try:
        if not pdf_documents: 
            print("Aucun PDF trouvé dans", pdf_path)
        else:
            print(f"{len(pdf_documents)} fichiers trouvés dans le répertoire {pdf_path} \n") 
            for i, file in enumerate(pdf_documents, 1): 
                file_name = os.path.splitext(os.path.basename(file))[0] 
                print("="*50, "\n") 
                print(f"Fichier {i}/{len(pdf_documents)}") 
                print(f"Fichier : {file_name} \n") 
                if file_name == "code_du_travail":
                    chunk_and_insert_pdf_file(
                        client=client,
                        collection=code_travail_col_name,
                        embedding_function=embedding_function,
                        file_path=file, 
                        separators=code_travail_separators)
                    print(f"{file_name} ajouté avec succès à la collection {code_travail_col_name}")
                elif file_name == "IDCC_liste":
                    chunk_and_insert_pdf_file(
                        client=client,
                        collection=idcc_ape_col_name,
                        embedding_function=embedding_function,
                        file_path=file, 
                        separators=idcc_separators)
                    print(f"{file_name} ajouté avec succès à la collection {idcc_ape_col_name}")
            reactivate_indexing(client = client, collection_name = idcc_ape_col_name)
            reactivate_indexing(client = client, collection_name = code_travail_col_name)
        return 1
    
    except Exception as e:
        print(f"Erreur de traitement : {e}")
        print("\n", "="*50, "\n")
        return 0

# ================================================================================================================

def ingest_idcc_ape_excel(files_path: str = files_path) -> int:
    
    excel_documents = load_and_read_excel_files(files_path)
    idcc_ape_col_name = "idcc_ape_collection"
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
    embedding_function = ef.OllamaEmbeddingFunction(model_name=embedding_model)
    
    create_collection(client=client, collection_name = idcc_ape_col_name, embedding_function=embedding_function)
    disable_indexing(client = client, collection_name = idcc_ape_col_name)
    

    try:
        # Filtrer les documents Excel
        idcc_excel_doc = [d for d in excel_documents if d["file"] == "IDCC_APE.xls"]
        ape_idcc_separators = ["\n"]
        
        if not idcc_excel_doc: 
            print("Aucun fichier Excel trouvé dans", files_path)
            return 0
        else:
            file_name = idcc_excel_doc[0]["file"]
            file_text = idcc_excel_doc[0]["text"]
            print("="*50, "\n") 
            print(f"Fichier : {file_name} \n") 
            
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
                print(f"Insertion réussie pour {file_name} ({len(ape_idcc_chunks)} chunks)")
                disable_indexing(client = client, collection_name = idcc_ape_col_name)
                return 1
            else:
                print(f"Erreur lors de l'insertion pour {file_name}")
                return 0
                
    except Exception as e:
        print(f"Erreur de traitement : {e}")
        print("\n", "="*50, "\n")
        return 0


    
# ================================================================================================================

if __name__ == "__main__":
    
    # ingestion des données :
    ingest_idcc_code_travail()
    ingest_idcc_ape_excel()
