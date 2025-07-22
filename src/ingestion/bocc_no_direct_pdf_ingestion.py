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
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "paraphrase-multilingual:278m-mpnet-base-v2-fp16")
files_path = os.path.join("data/BOCC_no_pdf_direct_link/")

# ==============================================================================================================

def extract_bocc_no_direct_metadata_expert(file_name: str) -> Dict[str, Optional[str]]:
    """
    Extracteur expert de métadonnées pour les bulletins officiels BOCC sans lien direct
    Patterns adaptés pour ce type de fichier spécifique
    """
    metadata = {
        "bulletin_number": None,
        "bulletin_date": None,
        "bulletin_year": None,
        "bulletin_type": "BOCC_no_direct_pdf",
        "source_type": "bulletin_scraped"
    }
    
    # Patterns spécifiques pour les fichiers sans lien direct
    # Ces fichiers peuvent avoir des noms différents après scraping
    bulletin_patterns = [
        r'n°\s*(\d{4}[_\-]\d+)',  # Format standard
        r'bulletin[_\-](\d{4})[_\-](\d+)',  # Format alternatif
        r'bocc[_\-](\d{4})[_\-](\d+)',  # Format BOCC
        r'(\d{4})[_\-](\d{1,3})',  # Format simplifié année-numéro
        r'num[eé]ro[_\s]*(\d{4}[_\-]\d+)'  # Variations "numéro"
    ]
    
    # Patterns pour dates avec plus de variabilité
    date_patterns = [
        r'du\s*(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})',  # Format: du 15_01_2024
        r'date[_\-](\d{1,2})[_\-](\d{1,2})[_\-](\d{4})',  # Format: date_15_01_2024
        r'(\d{1,2})[_\-](\d{1,2})[_\-](\d{4})',  # Format: 15_01_2024
        r'(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})',  # Format: 2024_01_15
        r'(\d{8})',  # Format: 20240115
    ]
    
    # Extraction du numéro de bulletin
    for pattern in bulletin_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                bulletin_number = match.group(1)
            else:
                # Cas où on a année et numéro séparés
                bulletin_number = f"{match.group(1)}_{match.group(2)}"
            
            metadata["bulletin_number"] = bulletin_number
            
            # Extraire l'année
            year_match = re.search(r'(\d{4})', bulletin_number)
            if year_match:
                metadata["bulletin_year"] = year_match.group(1)
            break
    
    # Extraction de la date avec gestion des formats multiples
    for pattern in date_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            try:
                if len(match.groups()) == 1:
                    # Format YYYYMMDD
                    date_str = match.group(1)
                    if len(date_str) == 8:
                        parsed_date = datetime.strptime(date_str, "%Y%m%d")
                        metadata["bulletin_date"] = parsed_date
                        metadata["bulletin_year"] = str(parsed_date.year)
                        break
                        
                elif len(match.groups()) == 3:
                    # Format avec jour, mois, année
                    day, month, year = match.groups()
                    
                    # Essayer différents ordres
                    date_combinations = [
                        (day, month, year),     # DD/MM/YYYY
                        (year, month, day),     # YYYY/MM/DD
                        (month, day, year),     # MM/DD/YYYY
                    ]
                    
                    for d, m, y in date_combinations:
                        try:
                            parsed_date = datetime(int(y), int(m), int(d))
                            metadata["bulletin_date"] = parsed_date
                            metadata["bulletin_year"] = str(parsed_date.year)
                            break
                        except ValueError:
                            continue
                    
                    if metadata["bulletin_date"]:
                        break
                        
            except Exception as e:
                print(f"Erreur parsing date dans {file_name}: {e}")
                continue
    
    return metadata

def get_bocc_no_direct_separators() -> List[str]:
    """
    Separators optimisés pour les bulletins BOCC sans lien direct
    Ces fichiers peuvent avoir une structure légèrement différente
    """
    base_separators = get_juridical_separators()
    
    # Separators spécifiques aux bulletins scrapés
    bocc_no_direct_separators = [
        # Structures typiques après scraping
        "\n\nTexte n° ",
        "\n\nTexte numéro ",
        "\n\nDocument n° ",
        "\n\nDocument numéro ",
        
        # Structures administratives
        "\n\nARRÊTÉ du ",
        "\n\nARRÊTÉ DU ",
        "\n\nDÉCRET n° ",
        "\n\nDÉCRET N° ",
        "\n\nCIRCULAIRE ",
        "\n\nINSTRUCTION ",
        "\n\nAVIS ",
        "\n\nDÉCISION ",
        "\n\nCOMMUNICATION ",
        
        # Structures d'extension
        "\n\nEXTENSION d'accord ",
        "\n\nEXTENSION de la convention ",
        "\n\nELARGISSEMENT ",
        "\n\nAGRÉMENT ",
        "\n\nDÉNONCIATION ",
        
        # Marqueurs de début/fin typiques
        "\n\nMinistère ",
        "\n\nDirection ",
        "\n\nService ",
        "\n\nBureau ",
        
        # Separators de contenu
        "\n\nVu ",
        "\n\nConsidérant ",
        "\n\nArrête ",
        "\n\nDécide ",
        
    ] + base_separators
    
    return bocc_no_direct_separators

def validate_bocc_no_direct_content(text: str) -> bool:
    """
    Validation simplifiée du contenu BOCC - Filtre seulement le bruit évident
    """
    if not text or len(text.strip()) < 40:
        return False
    
    # Filtre seulement le bruit évident
    noise_indicators = [
        r'^\s*\d+\s*$',  # Pages contenant seulement un numéro
        r'^\s*sommaire\s*$',  # Pages sommaire
        r'^\s*index\s*$',  # Pages index
        r'^\s*page\s*\d+\s*$',  # Page X
        r'^\s*imprimer\s*$',  # Bouton imprimer
        r'^\s*t[eé]l[eé]charger\s*$',  # Bouton télécharger
    ]
    
    text_lower = text.lower().strip()
    for noise in noise_indicators:
        if re.match(noise, text_lower):
            return False
    
    return True  # ✅ Accepte tout le reste


def preprocess_bocc_no_direct_text(text: str) -> str:
    """
    Preprocessing spécifique pour les bulletins sans lien direct
    """
    if not text:
        return ""
    
    # Nettoyage de base
    from src.utils import clean_juridical_text
    text = clean_juridical_text(text)
    
    # Corrections spécifiques aux bulletins scrapés
    
    # Nettoyer les artefacts de scraping
    text = re.sub(r'(?i)page\s+\d+\s+sur\s+\d+', '', text)
    text = re.sub(r'(?i)imprimer\s+cette\s+page', '', text)
    text = re.sub(r'(?i)retour\s+au\s+sommaire', '', text)
    text = re.sub(r'(?i)t[eé]l[eé]charger\s+le\s+pdf', '', text)
    
    # Standardiser les références administratives
    text = re.sub(r"(?i)minist[eè]re\s+du\s+travail,?\s+de\s+l['’]?emploi", 
              "Ministère du Travail, de l'Emploi", text)    
    
    # Standardiser les références de direction
    text = re.sub(r'(?i)direction\s+g[eé]n[eé]rale\s+du\s+travail', 
                  'Direction générale du travail', text)
    
    # Standardiser les structures d'arrêté
    text = re.sub(r'(?i)arr[eê]t[eé]\s+du\s+(\d{1,2})\s+(\w+)\s+(\d{4})', 
                  r'Arrêté du \1 \2 \3', text)
    
    # Standardiser les références aux textes
    text = re.sub(r'(?i)vu\s+le\s+code\s+du\s+travail', 'Vu le Code du travail', text)
    text = re.sub(r'(?i)vu\s+la\s+loi\s+n°\s*([0-9\-]+)', r'Vu la loi n° \1', text)
    
    # Corriger les numérotations
    text = re.sub(r'(?i)article\s+([A-Z]?\d+(?:[.\-]\d+)*)', r'Article \1', text)
    
    return text

def ingest_no_direct_pdf_bocc(pdf_path: str = files_path, client_host: str = client_host) -> int:
    """
    Ingestion expert des bulletins officiels BOCC sans lien direct PDF
    """
    collection_name = "bocc"
    separators = get_bocc_no_direct_separators()
    
    # Configuration chunks adaptée pour les bulletins scrapés
    chunk_size = 1200  # Légèrement plus petit pour les textes scrapés
    chunk_overlap = 300  # Plus de chevauchement pour compenser la fragmentation
    
    logs_dir = "logs/bocc_logs/no_direct_pdf/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_path = f"{logs_dir}log_{now_str}.log"
    
    with open(logfile_path, "w", encoding="utf-8") as logfile_handle:
        log_and_print("=" * 80, logfile_handle)
        log_and_print("INGESTION EXPERT BOCC - SANS LIEN DIRECT PDF", logfile_handle)
        log_and_print("=" * 80, logfile_handle)
        
        # Chargement des fichiers PDF
        try:
            pdf_documents = load_pdf_files(pdf_path)
            if pdf_documents:
                log_and_print(f"✓ {len(pdf_documents)} fichiers PDF chargés avec succès", logfile_handle)
            else:
                log_and_print(f"✗ Aucun fichier PDF trouvé dans {pdf_path}", logfile_handle)
                return 0
        except Exception as e:
            log_and_print(f"✗ Erreur chargement fichiers: {e}", logfile_handle)
            return 0
        
        # Configuration embedding
        try:
            embedding_function, active_model = get_embedding_function_with_fallback(logfile_handle)
            log_and_print(f"✓ Modèle embedding actif: {active_model}", logfile_handle)
        except Exception as e:
            log_and_print(f"✗ Impossible de charger les modèles d'embedding: {e}", logfile_handle)
            return 0
                
        # Connexion base de données
        try:
            client = get_qdrant_client(client_host, logfile=logfile_handle)
            create_collection_qdrant(client=client, collection_name=collection_name, 
                            embedding_function=embedding_function, logfile=logfile_handle)
            disable_indexing(client=client, collection_name=collection_name, logfile=logfile_handle)
            log_and_print(f"✓ Collection '{collection_name}' configurée", logfile_handle)
        except Exception as e:
            log_and_print(f"✗ Erreur configuration DB: {e}", logfile_handle)
            return 0
        
        # Traitement des fichiers
        success_count = 0
        error_count = 0
        
        try:
            for i, file_path in enumerate(pdf_documents, 1):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                
                log_and_print(f"\n{'='*60}", logfile_handle)
                log_and_print(f"TRAITEMENT {i}/{len(pdf_documents)}: {file_name}", logfile_handle)
                log_and_print(f"{'='*60}", logfile_handle)
                
                # Extraction des métadonnées avec l'extracteur expert adapté
                metadata = extract_bocc_no_direct_metadata_expert(file_name)
                
                # Log des métadonnées extraites
                log_and_print(f"📊 Métadonnées extraites:", logfile_handle)
                for key, value in metadata.items():
                    if value:
                        log_and_print(f"  • {key}: {value}", logfile_handle)
                
                # Validation spécifique
                if not metadata["bulletin_number"] and not metadata["bulletin_date"]:
                    log_and_print(f"⚠️  Aucune métadonnée critique extraite pour {file_name}", logfile_handle)
                    log_and_print(f"    Fichier traité comme bulletin générique", logfile_handle)
                
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
                        log_and_print(f"✅ {file_name} traité avec succès", logfile_handle)
                    else:
                        error_count += 1
                        log_and_print(f"❌ Échec traitement {file_name}", logfile_handle)
                        
                except Exception as e:
                    error_count += 1
                    log_and_print(f"❌ Erreur traitement {file_name}: {e}", logfile_handle)
            
            # Réactivation de l'indexation
            reactivate_indexing(client=client, collection_name=collection_name, logfile=logfile_handle)
            
            # Rapport final
            log_and_print(f"\n{'='*60}", logfile_handle)
            log_and_print(f"RAPPORT FINAL", logfile_handle)
            log_and_print(f"{'='*60}", logfile_handle)
            log_and_print(f"✅ Fichiers traités avec succès: {success_count}", logfile_handle)
            log_and_print(f"❌ Fichiers en erreur: {error_count}", logfile_handle)
            log_and_print(f"📊 Taux de succès: {success_count/(success_count+error_count)*100:.1f}%", logfile_handle)
            
            if success_count > 0:
                # Compter le total de chunks réellement insérés
                try:
                    collection_info = client.get_collection(collection_name)
                    total_chunks = getattr(collection_info, 'points_count', 0)
                    log_and_print(f"📊 Total chunks dans la collection: {total_chunks}", logfile_handle)
                    return total_chunks
                except:
                    return success_count
            else:
                return 0
            
        except Exception as e:
            log_and_print(f"❌ Erreur critique durant le traitement: {e}", logfile_handle)
            return 0

# ==============================================================================================================

if __name__ == "__main__":
    result = ingest_no_direct_pdf_bocc()
    if result:
        print("🎉 Ingestion BOCC sans lien direct terminée avec succès!")
    else:
        print("💥 Échec de l'ingestion BOCC sans lien direct")
