#!/usr/bin/env python3
"""
Script principal d'ingestion - Version simple et fonctionnelle
Lance toutes les ingestions des données juridiques françaises
"""

import sys
import os
from datetime import datetime

# Configuration du path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports des modules d'ingestion
from src.ingestion.idcc_code_travail_ingestion import ingest_idcc_code_travail, ingest_idcc_ape_excel
from src.ingestion.conventions_etendues_ingestion import ingestion_conventions_etendues
from src.ingestion.bocc_no_direct_pdf_ingestion import ingest_no_direct_pdf_bocc
from src.ingestion.bocc_direct_pdf_ingestion import ingest_direct_pdf_bocc

def main_ingest():
    """Lance toutes les ingestions des données juridiques"""
    
    print("🚀 INGESTION DONNÉES JURIDIQUES - DÉMARRAGE")
    print("=" * 60)
    print(f"📅 Début: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    
    # Liste des ingestions à exécuter
    ingestions = [
        ("⚖️  Code du Travail PDF", ingest_idcc_code_travail),
        ("📊 Correspondance IDCC-APE", ingest_idcc_ape_excel),
        ("🏢 Conventions Étendues", ingestion_conventions_etendues),
        ("📄 BOCC Liens Directs", ingest_direct_pdf_bocc),
        ("📋 BOCC Sans Liens Directs", ingest_no_direct_pdf_bocc)
    ]
    
    success_count = 0
    total_count = len(ingestions)
    
    # Exécution des ingestions
    for i, (name, func) in enumerate(ingestions, 1):
        print(f"\n[{i}/{total_count}] {name}")
        print("-" * 40)
        
        try:
            result = func()
            if result == 1:
                print(f"✅ {name} - SUCCÈS")
                success_count += 1
            else:
                print(f"❌ {name} - ÉCHEC")
        except Exception as e:
            print(f"❌ {name} - ERREUR: {e}")
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)
    print(f"✅ Réussies: {success_count}/{total_count}")
    print(f"❌ Échouées: {total_count - success_count}/{total_count}")
    print(f"🎯 Taux de succès: {(success_count/total_count)*100:.1f}%")
    print(f"⏰ Fin: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main_ingest()
    if success:
        print("\n🎉 INGESTION COMPLÈTE RÉUSSIE!")
    else:
        print("\n💥 INGESTION PARTIELLE - Vérifiez les erreurs ci-dessus")
