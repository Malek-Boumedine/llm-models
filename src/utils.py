from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import pandas as pd
from chromadb.api.models.Collection import Collection
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from chromadb.utils import embedding_functions
from typing import TextIO





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

def add_chunks_to_db(client: QdrantClient, collection: str, chunks: list, file_name: str, embedding_function, extra_metadata: dict = None) -> int:
    try:
        batch_size = 400  # Optimisé pour RTX 4060 Mobile
        idx = 1
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Générer les embeddings par batch
            vectors = embedding_function(batch_chunks)
            
            # S'assurer que c'est une liste native
            if hasattr(vectors, 'tolist'):
                vectors = vectors.tolist()
            
            # Créer les points
            points = []
            for j, (chunk, vector) in enumerate(zip(batch_chunks, vectors)):
                if not chunk or not chunk.strip():
                    continue  # Skip passages vides
                # Prépare les métadonnées
                payload = {
                    "text": chunk,
                    "source": file_name,
                    "chunk_id": idx,
                }
                if extra_metadata:
                    payload.update(extra_metadata)

                # Utilise PointStruct, ID auto ou UUID si besoin
                points.append(
                    PointStruct(
                        id=None,
                        vector=vector,
                        payload=payload))
                idx += 1

            # Insertion batch
            if points:
                client.upsert(collection_name=collection, points=points)
                print(f"Batch {i//batch_size + 1} inséré ({len(points)} points)")
        
        return 1
    except Exception as e:
        print(f"Erreur lors de l'ajout GPU : {e}")
        return 0

# ===========================================================================

# fonction qui lit le contenu d'un fichier, créé des chunks et l'envoie à la bdd

def chunk_and_insert_pdf_file(client : QdrantClient, collection : Collection, embedding_function : embedding_functions,  file_path : str, extra_metadata: dict, separators : list[str] = None, chunk_size : int = 1200, chunk_overlap : int = 200) -> int:

    # lecture du fichier pdf
    try :
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            rotation = page.get("/Rotate", 0)
            if rotation != 0 :
                # Tourner la page à l'endroit avant extraction (rotation négative)
                page.rotate(-rotation)
            texts.append(page.extract_text() or "")
        text = ''.join(texts)

        # création des chunks
        chunks = split_texts(text = text, separators=separators, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # envoie des chunks dans la db
        # file_name = file_path.split("/")[-1].replace(".pdf", "")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        chunk_succes = add_chunks_to_db(client, collection, chunks, file_name, embedding_function, extra_metadata)
        if chunk_succes :
            return 1
        else :
            return 0
    except Exception as e :
        print(f"Une erreur s'est produite. Erreur au niveau de la fonction {chunk_and_insert_pdf_file.__name__} : {e}")
        return 0

# ===========================================================================

# fonction qui permet d'écrire les logs

def log_and_print(message : str, logfile : TextIO) -> None : 
    print(message)
    logfile.write(message + "\n")
    logfile.flush()


# ===========================================================================


# ===========================================================================


# ===========================================================================


# ===========================================================================


# ===========================================================================


# ===========================================================================


# ===========================================================================

