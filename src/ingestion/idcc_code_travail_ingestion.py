import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import (load_and_read_excel_files, load_pdf_files, chunk_and_insert_pdf_file, 
                       split_texts, add_chunks_to_db, log_and_print, get_juridical_separators)

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
files_path = os.path.join("data/code_du_travail/")

# ==============================================================================================================

def extract_code_travail_metadata_expert(file_name: str) -> Dict[str, Optional[str]]:
    """
    Extracteur expert de métadonnées pour le Code du travail
    """
    metadata = {
        "document_type": "code_du_travail",
        "source_officielle": "Légifrance",
        "domaine_juridique": "Droit du travail",
        "version": None,
        "date_version": None
    }
    
    # Patterns pour identifier la version
    version_patterns = [
        r'(?i)version\s+(\d{4})',
        r'(?i)edition\s+(\d{4})',
        r'(?i)(\d{4})\s*edition',
        r'(?i)mise\s+[aà]\s+jour\s+(\d{4})'
    ]
    
    # Patterns pour les dates
    date_patterns = [
        r'(?i)du\s+(\d{1,2})\s+(\w+)\s+(\d{4})',
        r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})'
    ]
    
    # Extraction version
    for pattern in version_patterns:
        match = re.search(pattern, file_name)
        if match:
            metadata["version"] = match.group(1)
            break
    
    # Extraction date
    for pattern in date_patterns:
        match = re.search(pattern, file_name)
        if match:
            try:
                if len(match.groups()) == 3:
                    g1, g2, g3 = match.groups()
                    # Essayer différents formats
                    try:
                        parsed_date = datetime(int(g3), int(g2), int(g1))  # DD/MM/YYYY
                        metadata["date_version"] = parsed_date
                        break
                    except ValueError:
                        try:
                            parsed_date = datetime(int(g1), int(g2), int(g3))  # YYYY/MM/DD
                            metadata["date_version"] = parsed_date
                            break
                        except ValueError:
                            continue
            except:
                continue
    
    return metadata

def extract_idcc_ape_metadata_expert(file_name: str) -> Dict[str, Optional[str]]:
    """
    Extracteur expert de métadonnées pour les données IDCC-APE
    """
    metadata = {
        "document_type": "correspondance_idcc_ape",
        "source_officielle": "Ministère du Travail",
        "domaine_juridique": "Classifications professionnelles",
        "type_donnees": "correspondance",
        "format_source": "excel"
    }
    
    # Identifier le type de données
    if "cleaned" in file_name.lower():
        metadata["statut_donnees"] = "nettoyées"
    if "ape" in file_name.lower():
        metadata["contenu_principal"] = "codes_APE"
    if "idcc" in file_name.lower():
        metadata["contenu_principal"] = "codes_IDCC"
    
    return metadata

def get_code_travail_separators() -> List[str]:
    """
    Separators spécifiquement optimisés pour le Code du travail
    """
    base_separators = get_juridical_separators()
    
    # Separators spécifiques au Code du travail
    code_travail_separators = [
        # Structure hiérarchique du Code
        "\n\nPARTIE ",
        "\n\nPartie ",
        "\n\nLIVRE ",
        "\n\nLivre ",
        "\n\nTITRE ",
        "\n\nTitre ",
        "\n\nCHAPITRE ",
        "\n\nChapitre ",
        "\n\nSECTION ",
        "\n\nSection ",
        "\n\nSOUS-SECTION ",
        "\n\nSous-section ",
        "\n\nPARAGRAPHE ",
        "\n\nParagraphe ",
        "\n\nSOUS-PARAGRAPHE ",
        "\n\nSous-paragraphe ",
        
        # Articles du Code
        "\n\nARTICLE L",
        "\n\nArticle L",
        "\n\nARTICLE R",
        "\n\nArticle R",
        "\n\nARTICLE D",
        "\n\nArticle D",
        "\n\nART. L",
        "\n\nArt. L",
        "\n\nART. R",
        "\n\nArt. R",
        "\n\nART. D",
        "\n\nArt. D",
        
        # Structures spéciales
        "\n\nDISPOSITIONS GENERALES",
        "\n\nDispositions générales",
        "\n\nDISPOSITIONS COMMUNES",
        "\n\nDispositions communes",
        "\n\nDISPOSITIONS PARTICULIERES",
        "\n\nDispositions particulières",
        "\n\nDISPOSITIONS DIVERSES",
        "\n\nDispositions diverses",
        
        # Thématiques du Code du travail
        "\n\nCONTRAT DE TRAVAIL",
        "\n\nContrat de travail",
        "\n\nDUREE DU TRAVAIL",
        "\n\nDurée du travail",
        "\n\nSALAIRE",
        "\n\nSalaire",
        "\n\nCONGES PAYES",
        "\n\nCongés payés",
        "\n\nFORMATION PROFESSIONNELLE",
        "\n\nFormation professionnelle",
        "\n\nHYGIENE ET SECURITE",
        "\n\nHygiène et sécurité",
        "\n\nREPRESENTATION DU PERSONNEL",
        "\n\nReprésentation du personnel",
        "\n\nNEGOCIATION COLLECTIVE",
        "\n\nNégociation collective",
        "\n\nCONFLITS DU TRAVAIL",
        "\n\nConflits du travail",
        "\n\nCONTROLE DE L'APPLICATION",
        "\n\nContrôle de l'application",
        
    ] + base_separators
    
    return code_travail_separators

def get_idcc_ape_separators() -> List[str]:
    """
    Separators optimisés pour les données IDCC-APE
    """
    return [
        "\n\n",  # Séparateurs principaux
        "\n",    # Séparateurs de ligne
        ";",     # Séparateurs CSV
        ",",     # Séparateurs CSV alternatifs
        "\t",    # Tabulations
    ]

def validate_code_travail_content(text: str) -> bool:
    """
    Validation simplifiée du contenu Code du travail - Filtre seulement le bruit évident
    """
    if not text or len(text.strip()) < 50:
        return False
    
    # Filtre seulement le bruit évident
    noise_indicators = [
        r'^\s*\d+\s*$',  # Pages contenant seulement un numéro
        r'^\s*sommaire\s*$',  # Pages sommaire
        r'^\s*index\s*$',  # Pages index
        r'^\s*page\s*\d+\s*$',  # Page X
        r'^\s*table\s+des\s+mati[eè]res\s*$',  # Table des matières
        r'^\s*fin\s+du\s+document\s*$',  # Fin du document
    ]
    
    text_lower = text.lower().strip()
    for noise in noise_indicators:
        if re.match(noise, text_lower):
            return False
    
    return True  # ✅ Accepte tout le reste


def validate_idcc_ape_content(text: str) -> bool:
    """
    Validation simplifiée du contenu IDCC-APE - Filtre seulement le bruit évident
    """
    if not text or len(text.strip()) < 40:
        return False
    
    # Filtre seulement le bruit évident
    noise_indicators = [
        r'^\s*\d+\s*$',  # Pages contenant seulement un numéro
        r'^\s*sommaire\s*$',  # Pages sommaire
        r'^\s*index\s*$',  # Pages index
        r'^\s*page\s*\d+\s*$',  # Page X
        r'^\s*en-t[eê]te\s*$',  # En-têtes
        r'^\s*feuille\s*\d*\s*$',  # Feuille Excel
    ]
    
    text_lower = text.lower().strip()
    for noise in noise_indicators:
        if re.match(noise, text_lower):
            return False
    
    return True  # ✅ Accepte tout le reste


def preprocess_code_travail_text(text: str) -> str:
    """
    Preprocessing spécifique pour le Code du travail
    """
    if not text:
        return ""
    
    # Nettoyage de base
    from src.utils import clean_juridical_text
    text = clean_juridical_text(text)
    
    # Corrections spécifiques au Code du travail
    
    # Standardiser les références d'articles
    text = re.sub(r'(?i)article\s+([LRD])\s*\.?\s*(\d+(?:\-\d+)*)', r'Article \1. \2', text)
    text = re.sub(r'(?i)art\.\s*([LRD])\s*\.?\s*(\d+(?:\-\d+)*)', r'Article \1. \2', text)
    
    # Standardiser les structures hiérarchiques
    text = re.sub(r'(?i)livre\s+([IVXLCDM]+|\d+)', r'Livre \1', text)
    text = re.sub(r'(?i)titre\s+([IVXLCDM]+|\d+)', r'Titre \1', text)
    text = re.sub(r'(?i)chapitre\s+([IVXLCDM]+|\d+)', r'Chapitre \1', text)
    text = re.sub(r'(?i)section\s+([IVXLCDM]+|\d+)', r'Section \1', text)
    
    # Standardiser les références légales
    text = re.sub(r'(?i)code\s+du\s+travail', 'Code du travail', text)
    text = re.sub(r'(?i)code\s+de\s+la\s+s[eé]curit[eé]\s+sociale', 'Code de la sécurité sociale', text)
    
    # Standardiser les termes juridiques
    text = re.sub(r'(?i)contrat\s+[aà]\s+dur[eé]e\s+ind[eé]termin[eé]e', 'contrat à durée indéterminée', text)
    text = re.sub(r'(?i)contrat\s+[aà]\s+dur[eé]e\s+d[eé]termin[eé]e', 'contrat à durée déterminée', text)
    
    return text

def preprocess_idcc_ape_text(text: str) -> str:
    """
    Preprocessing spécifique pour les données IDCC-APE
    """
    # 🔥 FIX: Vérification robuste
    if not text or not isinstance(text, str) or not text.strip():
        return ""
    
    # Nettoyage de base
    from src.utils import clean_juridical_text
    text = clean_juridical_text(text)
    
    # Vérification après nettoyage
    if not text:
        return ""
    
    # Corrections spécifiques aux données IDCC-APE
    try:
        # Standardiser les codes IDCC
        text = re.sub(r"(?i)idcc\s*[:\-]?\s*(\d{4})", r"IDCC \1", text)
        
        # Standardiser les codes APE/NAF
        text = re.sub(r"(?i)ape\s*[:\-]?\s*(\d{2}\.?\d{2}[A-Z]?)", r"APE \1", text)
        text = re.sub(r"(?i)naf\s*[:\-]?\s*(\d{2}\.?\d{2}[A-Z]?)", r"NAF \1", text)
        
        # Standardiser les libellés
        text = re.sub(r"(?i)convention\s+collective", "Convention collective", text)
        text = re.sub(r"(?i)secteur\s+d[\'']?activit[eé]", "secteur d\'activité", text)
        
    except Exception as e:
        print(f"Erreur preprocessing IDCC-APE: {e}")
        return text  # Retourner le texte original si erreur
    
    return text


def ingest_idcc_code_travail(pdf_path: str = files_path) -> int:
    """
    Ingestion expert du Code du travail et des données IDCC PDF
    """
    code_travail_col_name = "code_travail_collection"
    idcc_ape_col_name = "idcc_ape_collection"
    code_travail_separators = get_code_travail_separators()
    idcc_separators = get_idcc_ape_separators()
    
    logs_dir = "logs/idcc_code_travail/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_path = f"{logs_dir}log_{now_str}.log"
    
    with open(logfile_path, "w", encoding="utf-8") as logfile_handle:
        log_and_print("=" * 80, logfile_handle)
        log_and_print("INGESTION EXPERT CODE DU TRAVAIL & IDCC PDF", logfile_handle)
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
        
        # Connexion base de données et création collections
        try:
            client = get_qdrant_client(client_host, logfile=logfile_handle)
            
            # Collection Code du travail
            create_collection_qdrant(client=client, collection_name=code_travail_col_name, 
                            embedding_function=embedding_function, logfile=logfile_handle)
            disable_indexing(client=client, collection_name=code_travail_col_name, logfile=logfile_handle)
            
            # Collection IDCC-APE
            create_collection_qdrant(client=client, collection_name=idcc_ape_col_name, 
                            embedding_function=embedding_function, logfile=logfile_handle)
            disable_indexing(client=client, collection_name=idcc_ape_col_name, logfile=logfile_handle)
            
            log_and_print(f"✓ Collections configurées", logfile_handle)
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
                
                try:
                    if file_name == "code_du_travail":
                        # Traitement Code du travail
                        metadata = extract_code_travail_metadata_expert(file_name)
                        log_and_print(f"📚 Traitement Code du travail", logfile_handle)
                        
                        result = chunk_and_insert_pdf_file(
                            client=client,
                            collection=code_travail_col_name,
                            embedding_function=embedding_function,
                            file_path=file_path,
                            extra_metadata=metadata,
                            separators=code_travail_separators,
                            chunk_size=1000,  # Plus grand pour les articles complets
                            chunk_overlap=250  # Overlap important pour les références croisées
                        )
                        
                        if result:
                            success_count += 1
                            log_and_print(f"✅ Code du travail traité avec succès", logfile_handle)
                        else:
                            error_count += 1
                            log_and_print(f"❌ Échec traitement Code du travail", logfile_handle)
                            
                    elif file_name == "IDCC_liste":
                        # Traitement liste IDCC
                        metadata = extract_idcc_ape_metadata_expert(file_name)
                        metadata["document_type"] = "liste_idcc_pdf"
                        log_and_print(f"📋 Traitement liste IDCC", logfile_handle)
                        
                        result = chunk_and_insert_pdf_file(
                            client=client,
                            collection=idcc_ape_col_name,
                            embedding_function=embedding_function,
                            file_path=file_path,
                            extra_metadata=metadata,
                            separators=idcc_separators,
                            chunk_size=800,  # Plus petit pour les listes
                            chunk_overlap=150
                        )
                        
                        if result:
                            success_count += 1
                            log_and_print(f"✅ Liste IDCC traitée avec succès", logfile_handle)
                        else:
                            error_count += 1
                            log_and_print(f"❌ Échec traitement liste IDCC", logfile_handle)
                    else:
                        log_and_print(f"⚠️  Fichier non reconnu: {file_name}", logfile_handle)
                        
                except Exception as e:
                    error_count += 1
                    log_and_print(f"❌ Erreur traitement {file_name}: {e}", logfile_handle)
            
            # Réactivation indexation
            reactivate_indexing(client=client, collection_name=code_travail_col_name, logfile=logfile_handle)
            reactivate_indexing(client=client, collection_name=idcc_ape_col_name, logfile=logfile_handle)
            
            # Rapport final
            log_and_print(f"\n{'='*60}", logfile_handle)
            log_and_print(f"RAPPORT FINAL PDF", logfile_handle)
            log_and_print(f"{'='*60}", logfile_handle)
            log_and_print(f"✅ Fichiers traités avec succès: {success_count}", logfile_handle)
            log_and_print(f"❌ Fichiers en erreur: {error_count}", logfile_handle)
            
            return 1 if success_count > 0 else 0
            
        except Exception as e:
            log_and_print(f"❌ Erreur critique durant le traitement: {e}", logfile_handle)
            return 0

def ingest_idcc_ape_excel(files_path: str = files_path, client_host: str = client_host) -> int:
    """
    Ingestion expert des données IDCC-APE Excel
    """
    idcc_ape_col_name = "idcc_ape_collection"
    ape_idcc_separators = get_idcc_ape_separators()
    
    logs_dir = "logs/idcc_code_travail/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_path = f"{logs_dir}log_excel_{now_str}.log"
    
    with open(logfile_path, "w", encoding="utf-8") as logfile_handle:
        log_and_print("=" * 80, logfile_handle)
        log_and_print("INGESTION EXPERT DONNÉES IDCC-APE EXCEL", logfile_handle)
        log_and_print("=" * 80, logfile_handle)
        
        # Chargement des fichiers Excel
        try:
            excel_documents = load_and_read_excel_files(files_path)
            if excel_documents:
                log_and_print(f"✓ {len(excel_documents)} fichiers Excel chargés avec succès", logfile_handle)
            else:
                log_and_print(f"✗ Aucun fichier Excel trouvé dans {files_path}", logfile_handle)
                return 0
        except Exception as e:
            log_and_print(f"✗ Erreur chargement fichiers Excel: {e}", logfile_handle)
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
            create_collection_qdrant(client=client, collection_name=idcc_ape_col_name, 
                            embedding_function=embedding_function, logfile=logfile_handle)
            disable_indexing(client=client, collection_name=idcc_ape_col_name, logfile=logfile_handle)
            log_and_print(f"✓ Collection '{idcc_ape_col_name}' configurée", logfile_handle)
        except Exception as e:
            log_and_print(f"✗ Erreur configuration DB: {e}", logfile_handle)
            return 0
        
        try:
            # Filtrer les documents Excel ciblés
            idcc_excel_docs = [d for d in excel_documents if "IDCC_APE_cleaned.xlsx" in d["file"]]
            
            if not idcc_excel_docs:
                log_and_print(f"✗ Fichier IDCC_APE_cleaned.xlsx non trouvé", logfile_handle)
                return 0
            
            for doc in idcc_excel_docs:
                file_name = doc["file"]
                file_text = doc["text"]
                
                log_and_print(f"\n{'='*60}", logfile_handle)
                log_and_print(f"TRAITEMENT EXCEL: {file_name}", logfile_handle)
                log_and_print(f"{'='*60}", logfile_handle)
                
                # Extraction métadonnées
                metadata = extract_idcc_ape_metadata_expert(file_name)
                
                # Log des métadonnées
                log_and_print(f"📊 Métadonnées extraites:", logfile_handle)
                for key, value in metadata.items():
                    if value:
                        log_and_print(f"  • {key}: {value}", logfile_handle)
                
                # Preprocessing du texte Excel
                if not file_text or not file_text.strip():
                    log_and_print(f"⚠️  Fichier Excel vide: {file_name}", logfile_handle)
                    continue
                processed_text = preprocess_idcc_ape_text(file_text)
                
                # Création des chunks optimisés pour les données tabulaires
                chunks = split_texts(
                    text=processed_text,
                    separators=ape_idcc_separators,
                    chunk_size=800,  # Plus grand pour les données tabulaires
                    chunk_overlap=200
                )
                
                # Filtrage des chunks valides
                valid_chunks = [chunk for chunk in chunks if validate_idcc_ape_content(chunk)]
                
                log_and_print(f"📋 {len(chunks)} chunks créés, {len(valid_chunks)} valides", logfile_handle)
                
                if valid_chunks:
                    success = add_chunks_to_db(
                        client=client,
                        collection=idcc_ape_col_name,
                        chunks=valid_chunks,
                        file_name=file_name,
                        embedding_function=embedding_function,
                        extra_metadata=metadata
                    )
                    
                    if success:
                        log_and_print(f"✅ Insertion réussie pour {file_name} ({len(valid_chunks)} chunks)", logfile_handle)
                    else:
                        log_and_print(f"❌ Erreur lors de l'insertion pour {file_name}", logfile_handle)
                        return 0
                else:
                    log_and_print(f"⚠️  Aucun chunk valide pour {file_name}", logfile_handle)
                    
            # Réactivation indexation
            reactivate_indexing(client=client, collection_name=idcc_ape_col_name, logfile=logfile_handle)
            
            # Rapport final
            log_and_print(f"\n{'='*60}", logfile_handle)
            log_and_print(f"RAPPORT FINAL EXCEL", logfile_handle)
            log_and_print(f"{'='*60}", logfile_handle)
            log_and_print(f"✅ Ingestion IDCC-APE Excel terminée avec succès", logfile_handle)
            
            return 1
            
        except Exception as e:
            log_and_print(f"❌ Erreur critique durant le traitement Excel: {e}", logfile_handle)
            return 0

# ================================================================================================================

if __name__ == "__main__":
    print("🚀 Démarrage ingestion Code du travail & IDCC-APE")
    
    # Ingestion des PDF
    result_pdf = ingest_idcc_code_travail()
    if result_pdf:
        print("✅ Ingestion PDF terminée avec succès")
    else:
        print("❌ Échec ingestion PDF")
    
    # Ingestion des Excel
    result_excel = ingest_idcc_ape_excel()
    if result_excel:
        print("✅ Ingestion Excel terminée avec succès")
    else:
        print("❌ Échec ingestion Excel")
    
    # Rapport final
    if result_pdf and result_excel:
        print("🎉 Ingestion complète terminée avec succès!")
    else:
        print("💥 Échec partiel ou total de l'ingestion")
