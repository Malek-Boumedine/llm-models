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
files_path = os.path.join("data/conventions_etendues/")

# ==============================================================================================================

def extract_convention_metadata_expert(file_name: str) -> Dict[str, Optional[str]]:
    """
    Extracteur expert de métadonnées pour les conventions collectives étendues
    Patterns optimisés pour identifier IDCC, secteur, et caractéristiques
    """
    metadata = {
        "idcc": None,
        "nom_convention": None,
        "secteur_activite": None,
        "type_convention": "convention_etendue",
        "niveau_convention": None,
        "date_signature": None
    }
    
    # Patterns robustes pour extraction IDCC
    idcc_patterns = [
        r'idcc\s*[:\-]?\s*n?[°]?\s*(\d{4})',  # IDCC: 1234 ou IDCC n°1234
        r'idcc\s*[:\-]?\s*(\d{4})',           # IDCC 1234
        r'(?:^|\s)(\d{4})\s*[:\-]?\s*idcc',   # 1234 IDCC
        r'convention\s+collective\s+n?[°]?\s*(\d{4})',  # Convention collective n°1234
        r'ccn\s*[:\-]?\s*(\d{4})',            # CCN 1234
        r'code\s+idcc\s*[:\-]?\s*(\d{4})'     # Code IDCC 1234
    ]
    
    # Patterns pour identifier le secteur d'activité
    secteur_patterns = [
        # Secteurs spécifiques
        (r'(?i)batiment|construction|btp|travaux\s+publics', 'BTP'),
        (r'(?i)metallurgie|m[eé]tallurgie|siderurgie|acier', 'Métallurgie'),
        (r'(?i)transport|logistique|routier|ferroviaire', 'Transport'),
        (r'(?i)commerce|commercial|n[eé]goce|distribution', 'Commerce'),
        (r'(?i)industrie|industriel|fabrication|production', 'Industrie'),
        (r'(?i)service|tertiaire|conseil|consultance', 'Services'),
        (r'(?i)sante|sant[eé]|medical|hopital|clinique', 'Santé'),
        (r'(?i)enseignement|education|formation|ecole', 'Éducation'),
        (r'(?i)banque|bancaire|finance|assurance|credit', 'Finance'),
        (r'(?i)hotel|restauration|tourism|cafeteria', 'Hôtellerie-Restauration'),
        (r'(?i)agriculture|agricole|elevage|forestier', 'Agriculture'),
        (r'(?i)textile|habillement|cuir|chaussure', 'Textile'),
        (r'(?i)chimie|chimique|pharmacie|cosmetique', 'Chimie'),
        (r'(?i)energie|electrique|gaz|petrole|nucleaire', 'Énergie'),
        (r'(?i)informatique|numerique|digital|software', 'Informatique')
    ]
    
    # Patterns pour niveau de convention
    niveau_patterns = [
        (r'(?i)nationale|national', 'national'),
        (r'(?i)regionale|regional', 'régional'),
        (r'(?i)departementale|departemental', 'départemental'),
        (r'(?i)locale|local', 'local'),
        (r'(?i)interprofessionnel', 'interprofessionnel'),
        (r'(?i)branche|sectoriel', 'branche')
    ]
    
    # Patterns pour date de signature
    date_patterns = [
        r'(?i)du\s+(\d{1,2})\s+(\w+)\s+(\d{4})',  # du 15 janvier 2024
        r'(?i)sign[eé]e?\s+le\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',  # signée le 15/01/2024
        r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',  # 15/01/2024
    ]
    
    # Extraction IDCC
    for pattern in idcc_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            idcc_number = match.group(1)
            metadata["idcc"] = idcc_number
            break
    
    # Extraction secteur d'activité
    for pattern, secteur in secteur_patterns:
        if re.search(pattern, file_name, re.IGNORECASE):
            metadata["secteur_activite"] = secteur
            break
    
    # Extraction niveau de convention
    for pattern, niveau in niveau_patterns:
        if re.search(pattern, file_name, re.IGNORECASE):
            metadata["niveau_convention"] = niveau
            break
    
    # Extraction date de signature
    for pattern in date_patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            try:
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    # Convertir le mois textuel si nécessaire
                    mois_fr = {
                        'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
                        'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
                        'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
                    }
                    
                    if month.lower() in mois_fr:
                        month = mois_fr[month.lower()]
                    
                    parsed_date = datetime(int(year), int(month), int(day))
                    metadata["date_signature"] = parsed_date
                    break
            except ValueError:
                continue
    
    # Nom de convention (nettoyé)
    nom_convention = file_name
    # Supprimer les extensions et caractères spéciaux
    nom_convention = re.sub(r'\.pdf$', '', nom_convention, flags=re.IGNORECASE)
    nom_convention = re.sub(r'[_\-]+', ' ', nom_convention)
    nom_convention = re.sub(r'\s+', ' ', nom_convention).strip()
    
    metadata["nom_convention"] = nom_convention
    
    return metadata

def get_convention_specific_separators() -> List[str]:
    """
    Separators spécifiquement optimisés pour les conventions collectives
    """
    base_separators = get_juridical_separators()
    
    # Separators spécifiques aux conventions collectives
    convention_separators = [
        # Structures principales des conventions
        "\n\nTITRE ",
        "\n\nTitre ",
        "\n\nCHAPITRE ",
        "\n\nChapitre ",
        "\n\nSECTION ",
        "\n\nSection ",
        "\n\nSOUS-SECTION ",
        "\n\nSous-section ",
        
        # Articles spécifiques
        "\n\nARTICLE ",
        "\n\nArticle ",
        "\n\nArt. ",
        "\n\nArt ",
        
        # Structures spéciales des conventions
        "\n\nPREAMBULE",
        "\n\nPréambule",
        "\n\nDISPOSITIONS GENERALES",
        "\n\nDispositions générales",
        "\n\nDISPOSITIONS PARTICULIERES",
        "\n\nDispositions particulières",
        
        # Classifications et salaires
        "\n\nCLASSIFICATION",
        "\n\nClassification",
        "\n\nGRILLE DE SALAIRES",
        "\n\nGrille de salaires",
        "\n\nSALAIRES",
        "\n\nSalaires",
        "\n\nREMUNERATION",
        "\n\nRémunération",
        
        # Temps de travail
        "\n\nDUREE DU TRAVAIL",
        "\n\nDurée du travail",
        "\n\nTEMPS DE TRAVAIL",
        "\n\nTemps de travail",
        "\n\nHORAIRES",
        "\n\nHoraires",
        "\n\nCONGES",
        "\n\nCongés",
        
        # Formation et carrière
        "\n\nFORMATION PROFESSIONNELLE",
        "\n\nFormation professionnelle",
        "\n\nPROMOTION",
        "\n\nPromotion",
        "\n\nEVOLUTION PROFESSIONNELLE",
        "\n\nÉvolution professionnelle",
        
        # Protection sociale
        "\n\nPROTECTION SOCIALE",
        "\n\nProtection sociale",
        "\n\nPREVOYANCE",
        "\n\nPrévoyance",
        "\n\nRETRAITE",
        "\n\nRetraite",
        "\n\nMUTUELLE",
        "\n\nMutuelle",
        
        # Conditions de travail
        "\n\nHYGIENE ET SECURITE",
        "\n\nHygiène et sécurité",
        "\n\nCONDITIONS DE TRAVAIL",
        "\n\nConditions de travail",
        
        # Représentation du personnel
        "\n\nREPRESENTATION DU PERSONNEL",
        "\n\nReprésentation du personnel",
        "\n\nDROIT SYNDICAL",
        "\n\nDroit syndical",
        
        # Annexes
        "\n\nANNEXE ",
        "\n\nAnnexe ",
        "\n\nAVENANT ",
        "\n\nAvenant ",
        "\n\nACCORD ",
        "\n\nAccord ",
        
    ] + base_separators
    
    return convention_separators

def validate_convention_content(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return False
    
    # Filtre seulement le bruit évident
    noise_indicators = [
        r'^\s*\d+\s*$',  # Pages contenant seulement un numéro
        r'^\s*sommaire\s*$',  # Pages sommaire
        r'^\s*index\s*$',  # Pages index
    ]
    
    text_lower = text.lower().strip()
    for noise in noise_indicators:
        if re.match(noise, text_lower):
            return False
    
    return True


def preprocess_convention_text(text: str) -> str:
    """
    Preprocessing spécifique pour les conventions collectives
    """
    if not text:
        return ""
    
    # Nettoyage de base
    from src.utils import clean_juridical_text
    text = clean_juridical_text(text)
    
    # Corrections spécifiques aux conventions collectives
    
    # Standardiser les références aux conventions
    text = re.sub(r'(?i)convention\s+collective\s+nationale', 
                  'Convention collective nationale', text)
    text = re.sub(r'(?i)ccn\s*n?[°]?\s*(\d+)', r'CCN n° \1', text)
    
    # Standardiser les codes IDCC
    text = re.sub(r'(?i)idcc\s*[:\-]?\s*n?[°]?\s*(\d{4})', r'IDCC \1', text)
    
    # Standardiser les références aux articles
    text = re.sub(r'(?i)article\s+([A-Z]?\d+(?:[.\-]\d+)*)', r'Article \1', text)
    text = re.sub(r'(?i)art\.\s*([A-Z]?\d+(?:[.\-]\d+)*)', r'Article \1', text)
    
    # Standardiser les structures de titre
    text = re.sub(r'(?i)titre\s+([IVXLCDM]+|\d+)', r'Titre \1', text)
    text = re.sub(r'(?i)chapitre\s+([IVXLCDM]+|\d+)', r'Chapitre \1', text)
    text = re.sub(r'(?i)section\s+([IVXLCDM]+|\d+)', r'Section \1', text)
    
    # Standardiser les références salariales
    text = re.sub(r'(?i)salaire\s+minimum', 'Salaire minimum', text)
    text = re.sub(r'(?i)coefficient\s+(\d+)', r'Coefficient \1', text)
    text = re.sub(r'(?i)niveau\s+([IVXLCDM]+|\d+)', r'Niveau \1', text)
    
    # Standardiser les durées
    text = re.sub(r'(?i)(\d+)\s*h\s*(\d+)', r'\1h\2', text)  # 35 h 00 -> 35h00
    text = re.sub(r'(?i)(\d+)\s*heures?', r'\1 heures', text)
    
    # Standardiser les congés
    text = re.sub(r'(?i)cong[eé]s?\s+pay[eé]s?', 'Congés payés', text)
    text = re.sub(r'(?i)cong[eé]s?\s+de\s+formation', 'Congés de formation', text)
    
    return text

def ingestion_conventions_etendues(pdf_path: str = files_path, client_host: str = client_host) -> int:
    """
    Ingestion expert des conventions collectives étendues
    """
    collection_name = "conventions_etendues"
    separators = get_convention_specific_separators()
    
    # Configuration chunks optimisée pour les conventions
    chunk_size = 800  # Plus grand pour capturer les articles complets
    chunk_overlap = 200  # Overlap important pour la cohérence des références
    
    logs_dir = "logs/conventions_etendues/"
    os.makedirs(logs_dir, exist_ok=True)
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    logfile_path = f"{logs_dir}log_{now_str}.log"
    
    with open(logfile_path, "w", encoding="utf-8") as logfile_handle:
        log_and_print("=" * 80, logfile_handle)
        log_and_print("INGESTION EXPERT CONVENTIONS COLLECTIVES ÉTENDUES", logfile_handle)
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
            # disable_indexing(client=client, collection_name=collection_name, logfile=logfile_handle)
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

                if i % 10 == 0:
                    log_and_print(f"🔄 Reconnexion client Qdrant après {i} fichiers", logfile_handle)
                    try:
                        client.close()  # Ferme l'ancienne connexion
                    except:
                        pass
                    client = get_qdrant_client(client_host, logfile=logfile_handle)  # Nouvelle connexion

                log_and_print(f"\n{'='*60}", logfile_handle)
                log_and_print(f"TRAITEMENT {i}/{len(pdf_documents)}: {file_name}", logfile_handle)
                log_and_print(f"{'='*60}", logfile_handle)
                
                # Extraction des métadonnées avec l'extracteur expert
                metadata = extract_convention_metadata_expert(file_name)
                
                # Log des métadonnées extraites
                log_and_print(f"📊 Métadonnées extraites:", logfile_handle)
                for key, value in metadata.items():
                    if value:
                        log_and_print(f"  • {key}: {value}", logfile_handle)
                
                # Validation spécifique
                if not metadata["idcc"]:
                    log_and_print(f"⚠️  Aucun IDCC extrait pour {file_name}", logfile_handle)
                if not metadata["secteur_activite"]:
                    log_and_print(f"⚠️  Secteur d'activité non identifié pour {file_name}", logfile_handle)
                
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
            # reactivate_indexing(client=client, collection_name=collection_name, logfile=logfile_handle)
            
            # Rapport final avec statistiques détaillées
            log_and_print(f"\n{'='*60}", logfile_handle)
            log_and_print(f"RAPPORT FINAL - CONVENTIONS COLLECTIVES", logfile_handle)
            log_and_print(f"{'='*60}", logfile_handle)
            log_and_print(f"✅ Fichiers traités avec succès: {success_count}", logfile_handle)
            log_and_print(f"❌ Fichiers en erreur: {error_count}", logfile_handle)
            if success_count + error_count > 0:
                log_and_print(f"📊 Taux de succès: {success_count/(success_count+error_count)*100:.1f}%", logfile_handle)
            
            return 1 if success_count > 0 else 0
            
        except Exception as e:
            log_and_print(f"❌ Erreur critique durant le traitement: {e}", logfile_handle)
            return 0

# ==============================================================================================================

if __name__ == "__main__":
    result = ingestion_conventions_etendues()
    if result:
        print("🎉 Ingestion conventions collectives étendues terminée avec succès!")
    else:
        print("💥 Échec de l'ingestion conventions collectives étendues")
