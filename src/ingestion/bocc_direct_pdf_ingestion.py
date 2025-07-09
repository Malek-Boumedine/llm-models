import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import  load_pdf_files, chunk_and_insert_pdf_file

import chromadb
from chromadb.utils import embedding_functions as ef
from chromadb.api.models.Collection import Collection
os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"



chroma_path = "BDD_TEST"
client = chromadb.PersistentClient(path=f"./{chroma_path}")

# embedding_model = "mxbai-embed-large"
# embedding_model = "paraphrase-multilingual:278m-mpnet-base-v2"  # Nom ollama du modèle
# embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)

# embedding_model = "BAAI/bge-base-en-v1.5"  # Nom HuggingFace du modèle
embedding_model = "paraphrase-multilingual-mpnet-base-v2"  # Nom HuggingFace du modèle
embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name = embedding_model)  # tres rapide mais moin adapté au domaine juridique. bon pour les tests


# ==============================================================================================================

# script_dir = os.path.dirname(os.path.abspath(__file__))
# bocc_pdf_path = os.path.join(script_dir, "../1.scraping_data/data/BOCC_pdf_direct_link/")
bocc_pdf_path = os.path.join("../1.scraping_data/data/BOCC_pdf_direct_link/")

# ==============================================================================================================

# chunk et embedding de tous les bulletins officiels des conventions étendues

collection_name = "bocc"
separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]

def ingest_direct_pdf_bocc(pdf_path : str = bocc_pdf_path, collection_name : str = collection_name, embedding_model : str = embedding_model, separators : list[str] = separators) -> int :
    
    bocc_pdf_documents = load_pdf_files(pdf_path)
    embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name = embedding_model)
    bocc_collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    
    try :
        if not bocc_pdf_documents : 
            print("Aucun PDF trouvé dans", pdf_path)
        else :
            print(f"{len(bocc_pdf_documents)} fichiers trouvés dans le répertoire {pdf_path} \n") 
            for i, file in enumerate(bocc_pdf_documents, 1) : 
                file_name = os.path.splitext(os.path.basename(file))[0] 
                print(f"Fichier {i}/{len(bocc_pdf_documents)}") 
                print("="*50, "\n") 
                print(f"Fichier : {file_name} \n") 
                
                chunk_and_insert_pdf_file(
                    collection = bocc_collection,
                    file_path = file, 
                    separators = separators, )
        return 1
    
    except Exception as e :
        print(f"Erreur de traitement : {e}")
    print("\n","="*50,"\n")
    return 0

# ==============================================================================================================

if __name__ == "__main__" : 
    
    ingest_direct_pdf_bocc()








