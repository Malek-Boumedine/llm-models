from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_and_read_excel_files, load_pdf_files, chunk_and_insert_pdf_file, split_texts, add_chunks_to_db

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

files_path = os.path.join("../1.scraping_data/data/code_du_travail/")

pdf_documents = load_pdf_files(files_path)
excel_documents = load_and_read_excel_files(files_path)

# print([d["file"] for d in pdf_documents])
# ['IDCC_liste.pdf', 'code_du_travail.pdf']

# ================================================================================================================

# chunks pour les icdd avec correspondance ape

def ingest_idcc_code_travail(pdf_path: str, embedding_model: str,) -> int:
    
    code_travail_col_name = "code_travail_collection"
    idcc_ape_col_name = "code_travail_idcc_ape_collection"
    code_travail_separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
    idcc_separators = ["\n"]

    pdf_documents = load_pdf_files(pdf_path)
    embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    code_travail_collection = client.get_or_create_collection(name=code_travail_col_name, embedding_function=embedding_function)
    idcc_ape_collection = client.get_or_create_collection(name=idcc_ape_col_name, embedding_function=embedding_function)
    
    try:
        if not pdf_documents: 
            print("Aucun PDF trouvé dans", pdf_path)
        else:
            print(f"{len(pdf_documents)} fichiers trouvés dans le répertoire {pdf_path} \n") 
            for i, file in enumerate(pdf_documents, 1): 
                file_name = os.path.splitext(os.path.basename(file))[0] 
                print(f"Fichier {i}/{len(pdf_documents)}") 
                print("="*50, "\n") 
                print(f"Fichier : {file_name} \n") 
                if file_name == "code_du_travail":
                    chunk_and_insert_pdf_file(
                        collection=code_travail_collection,
                        file_path=file, 
                        separators=code_travail_separators
                    )
                elif file_name == "IDCC_liste":
                    chunk_and_insert_pdf_file(
                        collection=idcc_ape_collection,
                        file_path=file, 
                        separators=idcc_separators
                    )
        return 1
    except Exception as e:
        print(f"Erreur de traitement : {e}")
        print("\n", "="*50, "\n")
        return 0

# ================================================================================================================

def ingest_idcc_ape_excel() :
    idcc_ape_col_name = "idcc_ape_collection"
    idcc_ape_collection = client.get_or_create_collection(name = idcc_ape_col_name, embedding_function=embedding_function)
    idcc_excel_doc = [d for d in excel_documents if d["file"] == "IDCC_APE.xls"]
    ape_idcc_separators = ["\n"]
    if idcc_excel_doc : 
        file_name = idcc_excel_doc[0]["file"]
        file_text = idcc_excel_doc[0]["text"]
        ape_idcc_chunks = split_texts(
            text=file_text,
            separators=ape_idcc_separators,
            chunk_size=2000, 
            chunk_overlap=200)
        add_chunks_to_db(idcc_ape_collection, ape_idcc_chunks, file_name)
        print(f"Insertion réussie pour {file_name} ({len(ape_idcc_chunks)} chunks)")
        return 1
    else:
        print("Aucun fichier IDCC_APE.xls trouvé")
        return 0
    
# ================================================================================================================

if __name__ == "__main__":
    ingest_idcc_code_travail(files_path, embedding_model)
    ingest_idcc_ape_excel()


