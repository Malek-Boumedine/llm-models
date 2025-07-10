from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import pandas as pd
from chromadb.api.models.Collection import Collection




# ===========================================================================

# chargement des fichiers pdf d'un dossier

def load_pdf_files(path : str) -> list[str] :
    files_paths = []
    files_list = os.listdir(path)
    for file in files_list :
        if file.endswith(".pdf") :
            file_path = os.path.join(path, file)
            files_paths.append(file_path)
    return files_paths

# ===========================================================================

# chargement et lecture des fichiers excel d'un dossier

def load_and_read_excel_files(path : str)-> list[dict] :
    docs = []
    files_list = os.listdir(path)
    for file in files_list :
        if file.endswith(".xls") or file.endswith(".xlsx") :
            df_dict = pd.read_excel(os.path.join(path, file), sheet_name=None)
            text = ""
            for name, sheet in df_dict.items():
                text += sheet.to_csv(index=False)
            docs.append({"file" : file, "text" : text})
    return docs

# ===========================================================================

# création des chunks avant l'embedding

def split_texts(text : str, separators = None, chunk_size : int = 1000, chunk_overlap : int = 150) -> list :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators = separators
    )
    return splitter.split_text(text)

# ===========================================================================

# ajouter les chunks à la base de données

def add_chunks_to_db(collection : Collection, chunks : list, file_name: str) -> int :
    try :
        ids = [f"{file_name}_{i}" for i in range(1, len(chunks)+1)]
        metadatas = [{"source": file_name} for _ in range(len(chunks))]
        
        # collection.add(
        #     documents=chunks,
        #     metadatas=metadatas,
        #     ids=ids)
        
        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            collection.add(
                documents=chunks[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size])
        return 1
    except Exception as e :
        print(f"Erreur lors de l'ajout : {e}")
        return 0

# ===========================================================================

# fonction qui lit le contenu d'un fichier, créé des chunks et l'envoie à la bdd

def chunk_and_insert_pdf_file(collection : Collection,  file_path : str, separators : list[str] = None, chunk_size : int = 1200, chunk_overlap : int = 200) -> int:

    # lecture du fichier pdf
    try :
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        text = ''.join(texts)

        # création des chunks
        chunks = split_texts(text = text, separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # envoie des chunks dans la db
        # file_name = file_path.split("/")[-1].replace(".pdf", "")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        chunk_succes = add_chunks_to_db(collection, chunks, file_name)
        if chunk_succes :
            return 1
        else :
            return 0
    except Exception as e :
        print(f"Une erreur s'est produite : {e}")
        return 0







