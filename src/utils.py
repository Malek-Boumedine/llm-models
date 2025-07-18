from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import pandas as pd
from chromadb.api.models.Collection import Collection
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from chromadb.utils import embedding_functions
from typing import TextIO, List, Dict, Optional
import re
import unicodedata




# ===========================================================================
# PREPROCESSING SPECIFIQUE AU CONTENU JURIDIQUE
# ===========================================================================

def clean_juridical_text(text: str) -> str:
    """
    Nettoie le texte juridique pour améliorer la qualité des embeddings
    """
    if not text:
        return ""
    
    # Normaliser les caractères Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Supprimer les caractères de contrôle mais garder les sauts de ligne
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Corriger les espaces multiples mais préserver la structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    
    # Supprimer les sauts de ligne excessifs (plus de 2 consécutifs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Corriger les numérotations d'articles malformées
    text = re.sub(r'(?i)article\s*[:\-]?\s*([A-Z]?\d+(?:\-\d+)*)', r'Article \1', text)
    
    # Standardiser les références de chapitre
    text = re.sub(r'(?i)chapitre\s*[:\-]?\s*([IVX]+|\d+)', r'Chapitre \1', text)
    
    # Standardiser les sections
    text = re.sub(r'(?i)section\s*[:\-]?\s*([IVX]+|\d+)', r'Section \1', text)
    
    return text.strip()

def validate_juridical_chunk(chunk: str) -> bool:
    """
    Valide qu'un chunk contient du contenu juridique pertinent
    """
    if not chunk or len(chunk.strip()) < 50:
        return False
    
    # Vérifier qu'il y a du contenu substantiel (pas que des espaces/numéros)
    meaningful_content = re.sub(r'[\s\d\-\.\,\;\:\(\)\[\]]+', '', chunk)
    if len(meaningful_content) < 20:
        return False
    
    # Éviter les chunks qui sont principalement des métadonnées de PDF
    metadata_patterns = [
        r'^\s*page\s*\d+\s*$',
        r'^\s*\d+\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',
        r'^\s*table\s+des\s+matières\s*$'
    ]
    
    for pattern in metadata_patterns:
        if re.match(pattern, chunk.strip(), re.IGNORECASE):
            return False
    
    return True

# ===========================================================================
# SEPARATORS OPTIMISÉS POUR LE CONTENU JURIDIQUE
# ===========================================================================

def get_juridical_separators() -> List[str]:
    """
    Retourne les separators optimisés pour le contenu juridique français
    """
    return [
        # Structures juridiques principales
        "\n\nChapitre ",
        "\n\nSection ",
        "\n\nSous-section ",
        "\n\nArticle ",
        "\n\nArt. ",
        "\n\nArt ",
        
        # Structures de convention collective
        "\n\nTitre ",
        "\n\nAVENANT ",
        "\n\nACCORD ",
        "\n\nANNEXE ",
        
        # Structures de bulletin officiel
        "\n\nARRÊTÉ ",
        "\n\nDÉCRET ",
        "\n\nCIRCULAIRE ",
        "\n\nINSTRUCTION ",
        
        # Séparateurs de paragraphes
        "\n\n",
        "\n",
        
        # Séparateurs de phrases pour le contenu dense
        ". ",
        " ; ",
        " - "
    ]

# ===========================================================================
# FONCTIONS DE BASE AMÉLIORÉES
# ===========================================================================

def load_pdf_files(path: str) -> List[str]:
    """Charge tous les fichiers PDF d'un dossier"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le chemin {path} n'existe pas")
    
    files_paths = []
    files_list = os.listdir(path)
    
    for file in files_list:
        if file.endswith(".pdf"):
            file_path = os.path.join(path, file)
            if os.path.getsize(file_path) > 0:  # Vérifier que le fichier n'est pas vide
                files_paths.append(file_path)
    
    return sorted(files_paths)  # Tri pour reproductibilité


def get_embedding_function_with_fallback(logfile=None):
    """
    Crée une fonction d'embedding robuste avec fallback automatique
    """
    from chromadb.utils import embedding_functions as ef
    
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
    fallback_model = os.getenv("FALLBACK_MODEL", "paraphrase-multilingual-mpnet-base-v2")
    
    try:
        embedding_function = ef.OllamaEmbeddingFunction(model_name=embedding_model)
        active_model = embedding_model
    except Exception as e:
        if logfile:
            log_and_print(f"⚠️ Fallback vers {fallback_model}: {e}", logfile)
        else:
            print(f"⚠️ Fallback vers {fallback_model}: {e}")
        try:
            embedding_function = ef.OllamaEmbeddingFunction(model_name=fallback_model)
            active_model = fallback_model
        except Exception as e2:
            error_msg = f"✗ Erreur critique - Les deux modèles ont échoué: {e2}"
            if logfile:
                log_and_print(error_msg, logfile)
            else:
                print(error_msg)
            raise e2
    
    return embedding_function, active_model


def load_and_read_excel_files(path: str) -> List[Dict]:
    """Charge et lit les fichiers Excel d'un dossier"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le chemin {path} n'existe pas")
    
    docs = []
    files_list = os.listdir(path)
    
    for file in files_list:
        if file.endswith((".xls", ".xlsx")):
            try:
                file_path = os.path.join(path, file)
                df_dict = pd.read_excel(file_path, sheet_name=None)
                text = ""
                for name, sheet in df_dict.items():
                    text += f"\n=== Feuille: {name} ===\n"
                    text += sheet.to_csv(index=False)
                
                # Nettoyer le texte Excel
                text = clean_juridical_text(text)
                docs.append({"file": file, "text": text})
            except Exception as e:
                print(f"Erreur lors de la lecture de {file}: {e}")
                continue
    
    return docs


def split_texts(
    text: str, 
    separators: Optional[List[str]] = None, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[str]:
    """
    Divise le texte en chunks optimisés pour le contenu juridique
    """
    if separators is None:
        separators = get_juridical_separators()
    
    # Nettoyer le texte avant chunking
    text = clean_juridical_text(text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = splitter.split_text(text)
    
    # Filtrer et valider les chunks
    valid_chunks = []
    for chunk in chunks:
        if validate_juridical_chunk(chunk):
            # Nettoyer chaque chunk individuellement
            clean_chunk = clean_juridical_text(chunk)
            if clean_chunk:
                valid_chunks.append(clean_chunk)
    
    return valid_chunks


def add_chunks_to_db(
    client: QdrantClient, 
    collection: str, 
    chunks: List[str], 
    file_name: str, 
    embedding_function, 
    extra_metadata: Optional[Dict] = None
) -> int:
    """
    Ajoute les chunks à la base de données avec validation
    """
    if not chunks:
        print("Aucun chunk valide à insérer")
        return 0
    
    try:
        batch_size = 400
        idx = 1
        total_inserted = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Générer les embeddings par batch
            try:
                vectors = embedding_function(batch_chunks)
                
                # S'assurer que c'est une liste native
                if hasattr(vectors, 'tolist'):
                    vectors = vectors.tolist()
                
                # Validation des vecteurs
                if len(vectors) != len(batch_chunks):
                    print(f"Erreur: nombre de vecteurs ({len(vectors)}) != nombre de chunks ({len(batch_chunks)})")
                    continue
                
            except Exception as e:
                print(f"Erreur génération embeddings batch {i//batch_size + 1}: {e}")
                continue
            
            # Créer les points
            points = []
            for j, (chunk, vector) in enumerate(zip(batch_chunks, vectors)):
                if not chunk or not chunk.strip():
                    continue
                
                # Préparer les métadonnées avec validation
                payload = {
                    "text": chunk,
                    "source": file_name,
                    "chunk_id": idx,
                    "chunk_length": len(chunk),
                    "chunk_words": len(chunk.split())
                }
                
                if extra_metadata:
                    # Valider les métadonnées avant ajout
                    for key, value in extra_metadata.items():
                        if value is not None:
                            payload[key] = value
                
                points.append(
                    PointStruct(
                        id=idx,
                        vector=vector,
                        payload=payload
                    )
                )
                idx += 1
            
            # Insertion batch avec gestion d'erreurs
            if points:
                try:
                    client.upsert(collection_name=collection, points=points)
                    total_inserted += len(points)
                    print(f"Batch {i//batch_size + 1} inséré ({len(points)} chunks valides)")
                except Exception as e:
                    print(f"Erreur insertion batch {i//batch_size + 1}: {e}")
                    continue
        
        print(f"Total inséré: {total_inserted} chunks pour {file_name}")
        return 1 if total_inserted > 0 else 0
        
    except Exception as e:
        print(f"Erreur lors de l'ajout des chunks: {e}")
        return 0


def chunk_and_insert_pdf_file(
    client: QdrantClient, 
    collection: str, 
    embedding_function, 
    file_path: str, 
    extra_metadata: Optional[Dict] = None, 
    separators: Optional[List[str]] = None, 
    chunk_size: int = 1200, 
    chunk_overlap: int = 200
) -> int:
    """
    Lit un PDF, crée des chunks optimisés et les insère dans la DB
    """
    try:
        # Lecture du fichier PDF avec gestion d'erreurs
        reader = PdfReader(file_path)
        
        if len(reader.pages) == 0:
            print(f"Fichier PDF vide: {file_path}")
            return 0
        
        texts = []
        for page_num, page in enumerate(reader.pages):
            try:
                # Gestion de la rotation
                rotation = page.get("/Rotate", 0)
                if rotation != 0:
                    page.rotate(-rotation)
                
                page_text = page.extract_text() or ""
                if page_text.strip():  # Seulement si la page a du contenu
                    texts.append(page_text)
                    
            except Exception as e:
                print(f"Erreur extraction page {page_num + 1} de {file_path}: {e}")
                continue
        
        if not texts:
            print(f"Aucun texte extrait de {file_path}")
            return 0
        
        # Joindre tout le texte et nettoyer
        text = '\n'.join(texts)
        text = clean_juridical_text(text)
        
        if not text or len(text.strip()) < 100:
            print(f"Texte insuffisant après nettoyage: {file_path}")
            return 0
        
        # Utiliser les separators juridiques par défaut
        if separators is None:
            separators = get_juridical_separators()
        
        # Création des chunks
        chunks = split_texts(
            text=text, 
            separators=separators, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        if not chunks:
            print(f"Aucun chunk valide généré pour {file_path}")
            return 0
        
        # Nom du fichier pour les métadonnées
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Insertion dans la DB
        success = add_chunks_to_db(
            client=client,
            collection=collection,
            chunks=chunks,
            file_name=file_name,
            embedding_function=embedding_function,
            extra_metadata=extra_metadata
        )
        
        return success
        
    except Exception as e:
        print(f"Erreur dans chunk_and_insert_pdf_file pour {file_path}: {e}")
        return 0


def log_and_print(message: str, logfile: TextIO) -> None:
    """Écrit les logs avec gestion d'erreurs"""
    try:
        print(message)
        logfile.write(message + "\n")
        logfile.flush()
    except Exception as e:
        print(f"Erreur écriture log: {e}")
        print(message)  # Au moins afficher le message

