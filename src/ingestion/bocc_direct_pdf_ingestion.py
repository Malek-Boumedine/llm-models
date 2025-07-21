import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_pdf_files, chunk_and_insert_pdf_file, log_and_print, get_juridical_separators

from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from src.db.connection import get_qdrant_client, create_collection_qdrant, disable_indexing, reactivate_indexing
from datetime import datetime
import re
from typing import Dict, Optional, List
from src.utils import get_embedding_function_with_fallback

# ==============================================================================================================

os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"
load_dotenv()

client_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
files_path = os.path.join("data/BOCC_pdf_direct_link")


# ==============================================================================================================

def extract_bocc_metadata_expert(file_name: str) -> Dict[str, Optional[str]]:
    """
    Extracteur expert de métadonnées pour les bulletins officiels BOCC
    Patterns optimisés pour une extraction fiable
    """
    metadata = {
        "bulletin_number": None,
        "bulletin_date": None,
        "bulletin_year": None,
        "bulletin_type": "BOCC_direct_pdf"
    }
    
    # Patterns robustes pour extraction bulletin
    bulletin_patterns = [
        r'n°\s*(\d{4}[_\-]\d+)',  # Format principal: n° 2024_15
        r'n°\s*(\d{4}[_\-]\d{1,3})',  # Variations avec 1-3 chiffres
        r'numero\s*(\d{4}[_\-]\d+)',  # Alternative "numero"
        r'bulletin\s*(\d{4}[_\-]\d+)'  # Alternative "bulletin"
    ]
    
    # Patterns robustes pour extraction date
    date_patterns = [
        r'du\s*(\d{1,2}[_\-]\d{1,2}[_\-]\d{4})',  # Format principal: du 15_01_2024
        r'date\s*(\d{1,2}[_\-]\d{1,2}[_\-]\d{4})',  # Alternative "date"
        r'(\d{1,2}[_\-]\d{1,2}[_\-]\d{4})'  # Date seule
    ]
    
    # Extraction du numéro de bulletin
    for pattern in bulletin_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            bulletin_number = match.group(1)
            metadata["bulletin_number"] = bulletin_number
            # Extraire l'année du bulletin
            year_match = re.search(r'(\d{4})', bulletin_number)
            if year_match:
                metadata["bulletin_year"] = year_match.group(1)
            break
    
    # Extraction de la date
    for pattern in date_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Normaliser les séparateurs
                date_normalized = re.sub(r'[_\-]', '_', date_str)
                # Essayer différents formats
                date_formats = ["%d_%m_%Y", "%d_%m_%y", "%Y_%m_%d"]
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_normalized, fmt)
                        metadata["bulletin_date"] = parsed_date
                        if not metadata["bulletin_year"]:
                            metadata["bulletin_year"] = str(parsed_date.year)
                        break
                    except ValueError:
                        continue
                        
                if metadata["bulletin_date"]:
                    break
                    
            except Exception as e:
                print(f"Erreur parsing date {date_str}: {e}")
                continue
    
    return metadata

def get_bocc_specific_separators() -> List[str]:
    """
    Separators spécifiquement optimisés pour les bulletins officiels
    """
    base_separators = get_juridical_separators()
    
    # Separators spécifiques aux bulletins officiels
    bocc_separators = [
        # Structures spécifiques BOCC
        "\n\nARRÊTÉ du ",
        "\n\nARRÊTÉ DU ",
        "\n\nDÉCRET n° ",
        "\n\nDÉCRET N° ",
        "\n\nCIRCULAIRE du ",
        "\n\nCIRCULAIRE DU ",
        "\n\nINSTRUCTION du ",
        "\n\nAVIS ",
        "\n\nCOMMUNICATION ",
        
        # Extensions spécifiques
        "\n\nEXTENSION ",
        "\n\nELARGISSEMENT ",
        "\n\nAGRÉMENT ",
        "\n\nDÉNONCIATION ",
        
        # Structures d'articles dans les bulletins
        "\n\nLe présent arrêté ",
        "\n\nLa présente décision ",
        "\n\nLe présent accord ",
        
        # Numérotation spécifique
        "\n\n1° ",
        "\n\n2° ",
        "\n\n3° ",
        "\n\n- ",
        
    ] + base_separators
    
    return bocc_separators

def validate_bocc_content(text: str) -> bool:
    """
    Validation spécifique du contenu BOCC
    """
    if not text or len(text.strip()) < 30:
        return False
    
    # Indicateurs de contenu BOCC valide
    bocc_indicators = [
        r'convention\s+collective',
        r'arr[eê]t[eé]',
        r'd[eé]cret',
        r'circulaire',
        r'extension',
        r'[eé]largissement',
        r'agr[eé]ment',
        r'bulletin\s+officiel',
        r'code\s+du\s+travail',
        r'dur[eé]e\s+du\s+travail',
        r'salaire',
        r'cong[eé]s',
        r'formation\s+professionnelle'
    ]
    
    # Vérifier la présence d'au moins un indicateur
    text_lower = text.lower()
    for indicator in bocc_indicators:
        if re.search(indicator, text_lower):
            return True
    
    return False

def preprocess_bocc_text(text: str) -> str:
    """
    Preprocessing spécifique pour les bulletins officiels
    """
    if not text:
        return ""
    
    # Nettoyage de base
    from src.utils import clean_juridical_text
    text = clean_juridical_text(text)
    
    # Corrections spécifiques aux bulletins officiels
    
    # Standardiser les références légales
    text = re.sub(r'(?i)code\s+du\s+travail\s*[,\s]*article\s*([A-Z]?\d+(?:\-\d+)*)', 
                  r'Code du travail, article \1', text)
    
    # Standardiser les dates dans le texte
    text = re.sub(r'(?i)arr[eê]t[eé]\s+du\s+(\d{1,2})\s+(\w+)\s+(\d{4})', 
                  r'Arrêté du \1 \2 \3', text)
    
    # Standardiser les références aux conventions
    text = re.sub(r'(?i)convention\s+collective\s+nationale\s+(?:du\s+)?(\d{1,2})\s+(\w+)\s+(\d{4})', 
                  r'Convention collective nationale du \1 \2 \3', text)
    
    # Corriger la numérotation des articles
    text = re.sub(r'(?i)art\.\s*([A-Z]?\d+(?:\-\d+)*)', r'Article \1', text)
    text = re.sub(r'(?i)art\s+([A-Z]?\d+(?:\-\d+)*)', r'Article \1', text)
    
    # Standardiser les codes IDCC
    text = re.sub(r'(?i)idcc\s*[:\-]?\s*n?\s*(\d+)', r'IDCC \1', text)
    
    return text

def ingest_direct_pdf_bocc(pdf_path: str = files_path, client_host: str = client_host) -> int:
    """
    Ingestion expert des bulletins officiels avec extraction de métadonnées optimisée
    """
    collection_name = "bocc"
    separators = get_bocc_specific_separators()
    
    # Configuration chunks optimisée pour BOCC
    chunk_size = 1500  # Plus grand pour capturer les articles complets
    chunk_overlap = 250  # Plus de chevauchement pour la cohérence juridique
    
    logs_dir = "logs/bocc_logs/direct_pdf/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_path = f"{logs_dir}log_{now_str}.log"
    
    with open(logfile_path, "w", encoding="utf-8") as logfile:
        log_and_print("=" * 80, logfile)
        log_and_print("INGESTION EXPERT BOCC - LIENS DIRECTS PDF", logfile)
        log_and_print("=" * 80, logfile)
        
        # Chargement des fichiers PDF
        try:
            pdf_documents = load_pdf_files(pdf_path)
            if pdf_documents:
                log_and_print(f"✓ {len(pdf_documents)} fichiers PDF chargés avec succès", logfile)
            else:
                log_and_print(f"✗ Aucun fichier PDF trouvé dans {pdf_path}", logfile)
                return 0
        except Exception as e:
            log_and_print(f"✗ Erreur chargement fichiers: {e}", logfile)
            return 0
        
        # Configuration embedding
        try:
            embedding_function, active_model = get_embedding_function_with_fallback(logfile)
            log_and_print(f"✓ Modèle embedding actif: {active_model}", logfile)
        except Exception as e:
            log_and_print(f"✗ Impossible de charger les modèles d'embedding: {e}", logfile)
            return 0
                
        # Connexion base de données
        try:
            client = get_qdrant_client(client_host, logfile=logfile)
            create_collection_qdrant(client=client, collection_name=collection_name, 
                            embedding_function=embedding_function, logfile=logfile)
            disable_indexing(client=client, collection_name=collection_name, logfile=logfile)
            log_and_print(f"✓ Collection '{collection_name}' configurée", logfile)
        except Exception as e:
            log_and_print(f"✗ Erreur configuration DB: {e}", logfile)
            return 0
        
        # Traitement des fichiers
        success_count = 0
        error_count = 0
        
        try:
            for i, file_path in enumerate(pdf_documents, 1):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                
                log_and_print(f"\n{'='*60}", logfile)
                log_and_print(f"TRAITEMENT {i}/{len(pdf_documents)}: {file_name}", logfile)
                log_and_print(f"{'='*60}", logfile)
                
                # Extraction des métadonnées avec l'extracteur expert
                metadata = extract_bocc_metadata_expert(file_name)
                
                # Log des métadonnées extraites
                log_and_print(f"📊 Métadonnées extraites:", logfile)
                for key, value in metadata.items():
                    if value:
                        log_and_print(f"  • {key}: {value}", logfile)
                
                # Validation des métadonnées critiques
                if not metadata["bulletin_number"] and not metadata["bulletin_date"]:
                    log_and_print(f"⚠️  Aucune métadonnée critique extraite pour {file_name}", logfile)
                
                # Traitement du fichier avec métadonnées
                try:
                    result = chunk_and_insert_pdf_file(
                        client=client,
                        collection=collection_name,
                        embedding_function=embedding_function,
                        file_path=file_path,
                        extra_metadata=metadata,  # 🔥 FIX CRITIQUE: Passage des métadonnées
                        separators=separators,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    if result:
                        success_count += 1
                        log_and_print(f"✅ {file_name} traité avec succès", logfile)
                    else:
                        error_count += 1
                        log_and_print(f"❌ Échec traitement {file_name}", logfile)
                        
                except Exception as e:
                    error_count += 1
                    log_and_print(f"❌ Erreur traitement {file_name}: {e}", logfile)
            
            # Réactivation de l'indexation
            reactivate_indexing(client=client, collection_name=collection_name, logfile=logfile)
            
            # Rapport final
            log_and_print(f"\n{'='*60}", logfile)
            log_and_print(f"RAPPORT FINAL", logfile)
            log_and_print(f"{'='*60}", logfile)
            log_and_print(f"✅ Fichiers traités avec succès: {success_count}", logfile)
            log_and_print(f"❌ Fichiers en erreur: {error_count}", logfile)
            log_and_print(f"📊 Taux de succès: {success_count/(success_count+error_count)*100:.1f}%", logfile)
            
            return 1 if success_count > 0 else 0
            
        except Exception as e:
            log_and_print(f"❌ Erreur critique durant le traitement: {e}", logfile)
            return 0

# ==============================================================================================================

if __name__ == "__main__":
    result = ingest_direct_pdf_bocc()
    if result:
        print("🎉 Ingestion BOCC direct PDF terminée avec succès!")
    else:
        print("💥 Échec de l'ingestion BOCC direct PDF")
