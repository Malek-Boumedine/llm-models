from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_excel_files, load_pdf_files, split_texts, add_chunks_to_db

import chromadb
from chromadb.utils import embedding_functions as ef
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
excel_documents = load_excel_files(files_path)

# print([d["file"] for d in pdf_documents])
# ['IDCC_liste.pdf', 'code_du_travail.pdf']

# ================================================================================================================

# chunks pour la liste des idcc

idcc_ape_col_name = "idcc_ape_collection"
idcc_ape_collection = client.get_or_create_collection(name = idcc_ape_col_name, embedding_function=embedding_function)

idcc_text = [d for d in pdf_documents if d["file"] == "IDCC_liste.pdf"]
idcc_separators = ["\n"]
if idcc_text:
    file_name = idcc_text[0]["file"]
    file_text = idcc_text[0]["text"]
    idcc_chunks = split_texts(
        text=file_text,
        separators=idcc_separators,
        chunk_size=1200,
        chunk_overlap=200)
    # print(len(idcc_chunks))
    
    # embedding :
    add_chunks_to_db(idcc_ape_collection, idcc_chunks, file_name)

# ================================================================================================================

# chunks pour les icdd avec correspondance ape

idcc_excel_doc = [d for d in excel_documents if d["file"] == "IDCC_APE.xls"]
ape_idcc_separators = ["\n"]
if idcc_excel_doc : 
    file_name = idcc_excel_doc[0]["file"]
    file_text = idcc_excel_doc[0]["text"]
    ape_idcc_chunks = split_texts(
        text=idcc_excel_doc[0]["text"],
        separators=ape_idcc_separators,
        chunk_size=2000, 
        chunk_overlap=200)
    
    # embedding :
    add_chunks_to_db(idcc_ape_collection, ape_idcc_chunks, file_name)
    

# ================================================================================================================

# chunks pour le code du travail

code_travail_col_name = "code_travail_collection"
code_travail_collection = client.get_or_create_collection(name = code_travail_col_name, embedding_function=embedding_function)

code_travail_text = [d for d in pdf_documents if d["file"] == "code_du_travail.pdf"]
code_travail_separators = custom_separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]
if code_travail_text:
    file_name = code_travail_text[0]["file"]
    file_text = code_travail_text[0]["text"]
    code_travail_chunks = split_texts(
        text=code_travail_text[0]["text"],
        separators=code_travail_separators,
        chunk_size=1200,
        chunk_overlap=200)
    
    # embedding :
    add_chunks_to_db(code_travail_collection, code_travail_chunks, file_name)
    
# ================================================================================================================


