from src.ingestion.bocc_direct_pdf_ingestion import ingest_direct_pdf_bocc
from src.ingestion.bocc_no_direct_pdf_ingestion import ingest_no_direct_pdf_bocc
from src.ingestion.conventions_etendues_ingestion import ingestion_conventions_etendues
from src.ingestion.idcc_code_travail_ingestion import ingest_idcc_ape_excel, ingest_idcc_code_travail




def main():
    print("="*25 + "  DEBUT D'INGESTION DES DONNEES  " + "="*25 + "\n\n")
    
    print("ingest_idcc_ape_excel \n")
    print("-"*123)
    ingest_idcc_ape_excel()
    print("ingest_idcc_code_travail \n")
    print("-"*123)
    ingest_idcc_code_travail()
    print("ingestion_conventions_etendues \n")
    print("-"*123)
    ingestion_conventions_etendues()
    print("ingest_no_direct_pdf_bocc \n")
    print("-"*123)
    ingest_no_direct_pdf_bocc()
    print("ingest_direct_pdf_bocc \n")
    print("-"*123)
    ingest_direct_pdf_bocc()


if __name__ == "__main__":
    main()
