import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import load_excel_files, load_pdf_files, split_texts, add_chunks_to_db

import chromadb
from chromadb.utils import embedding_functions as ef
os.environ["CHROMA_ENABLE_TELEMETRY"] = "False"




chroma_path = "BDD_TEST"
client = chromadb.PersistentClient(path=f"./{chroma_path}")

# embedding_model = "mxbai-embed-large"
# embedding_model = "paraphrase-multilingual:278m-mpnet-base-v2"  # Nom ollama du modèle
# embedding_function = ef.OllamaEmbeddingFunction(model_name = embedding_model)

# embedding_model = "BAAI/bge-base-en-v1.5"  # Nom HuggingFace du modèle
embedding_model = "paraphrase-multilingual-mpnet-base-v2"  # Nom HuggingFace du modèle
embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name = embedding_model)  # tres rapide mais moin adapté au domaine juridique. bon pour les tests


# ==============================================================================================================

files_path = os.path.join("../1.scraping_data/data/conventions_etendues/")
pdf_documents = load_pdf_files(files_path)

print(len(pdf_documents))

# ==============================================================================================================

# chunk et embedding de toutes les conventions étendues

collection_name = "conventions_etendues"
conventions_etendues_collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
separators = ["\n\n", "\nArticle ", "\nChapitre", "\nSection", "\n", " "]

for i, document in enumerate(pdf_documents, 1) : 
    print(f"Document {i}/{len(pdf_documents)}")
    file_name = document["file"]
    file_text = document["text"]
    document_chunk = split_texts(
        text=file_text, 
        separators=separators, chunk_size=1200, 
        chunk_overlap=200
    )
    add_chunks_to_db(collection=conventions_etendues_collection, chunks=document_chunk, file_name=file_name)


# ==============================================================================================================


