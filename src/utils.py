from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import pandas as pd
from chromadb.api.models.Collection import Collection




# chargement des fichiers pdf d'un dossier

def load_pdf_files(path) :
    docs = []
    files_list = os.listdir(path)
    for file in files_list :
        if file.endswith(".pdf") :
            reader = PdfReader(os.path.join(path, file))
            text = ""
            for page in reader.pages :
                text += page.extract_text() or ""
            docs.append({"file" : file, "text" : text})
    return docs
            

# chargement des fichiers excel d'un dossier

def load_excel_files(path) :
    docs = []
    files_list = os.listdir(path)
    for file in files_list :
        if file.endswith(".xls") or file.endswith(".xlsx") :
            df = pd.read_excel(os.path.join(path, file))
            text = ""
            for name, sheet in df.items() : 
                text += sheet.to_csv(index=False)
            docs.append({"file" : file, "text" : text})
    return docs

    
# création des chunks avant l'embedding

def split_texts(text, separators = None, chunk_size=1000, chunk_overlap=150) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        separators = separators
    )
    return splitter.split_text(text)

# ajouter les chunks à la base de données

def add_chunks_to_db(collection: Collection, chunks: list, file_name: str) -> None: 
    ids = [f"{file_name}_{i}" for i in range(1, len(chunks)+1)]
    metadatas = [{"source": "IDCC_APE.xls"} for _ in range(len(chunks))]
    # collection.add(
    #     documents=chunks, 
    #     metadatas=metadatas, 
    #     ids=ids)
    
    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    



