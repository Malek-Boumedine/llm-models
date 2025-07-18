from src.ingestion.bocc_direct_pdf_ingestion import ingest_direct_pdf_bocc
from src.ingestion.bocc_no_direct_pdf_ingestion import ingest_no_direct_pdf_bocc
from src.ingestion.conventions_etendues_ingestion import ingestion_conventions_etendues
from src.ingestion.idcc_code_travail_ingestion import ingest_idcc_ape_excel, ingest_idcc_code_travail


def main_ingest():
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


# API

# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from src.api.endpoints.auth_endpoint import router as auth_router
from src.api.endpoints.llm_chat_endpoint import router as llm_router

app = FastAPI(
    title="Master Agent LLM API", 
    description="API pour interactions avec le Master Agent spécialisé en droit du travail français", 
    version="1.0.0"
)

# Route d'accueil avec belle présentation
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Master Agent LLM API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .card { background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #3498db; }
            .endpoint { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #27ae60; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Master Agent LLM API</h1>
            <p>API spécialisée en droit du travail français</p>
            <p class="status">✅ Service actif</p>
        </div>
        
        <div class="card">
            <h2>📚 Documentation</h2>
            <p>Accédez à la documentation interactive :</p>
            <p>🔗 <a href="/docs" target="_blank">Swagger UI (/docs)</a></p>
            <p>🔗 <a href="/redoc" target="_blank">ReDoc (/redoc)</a></p>
        </div>
        
        <div class="card">
            <h2>🔐 Authentification</h2>
            <div class="endpoint">
                <p><span class="method">POST</span> <a href="/docs#/default/login_api_v1_auth_post">/api/v1/auth</a></p>
                <p>Connexion utilisateur (email + mot de passe)</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> <a href="/docs#/default/register_api_v1_register_post">/api/v1/register</a></p>
                <p>Création de compte (admin uniquement)</p>
            </div>
            <div class="endpoint">
                <p><span class="method">GET</span> <a href="/docs#/default/get_current_user_info_api_v1_me_get">/api/v1/me</a></p>
                <p>Informations utilisateur connecté</p>
            </div>
        </div>
        
        <div class="card">
            <h2>💬 Chat LLM</h2>
            <div class="endpoint">
                <p><span class="method">POST</span> <a href="/docs#/default/llm_chat_api_v1_llm_chat_post">/api/v1/llm_chat</a></p>
                <p>Poser une question au Master Agent (authentification requise)</p>
            </div>
        </div>
        
        <div class="card">
            <h2>ℹ️ Informations</h2>
            <p><strong>Version :</strong> 1.0.0</p>
            <p><strong>Spécialité :</strong> Droit du travail français</p>
            <p><strong>Agents disponibles :</strong> Droit général, Conventions collectives, Recherche web</p>
        </div>
        
        <div class="card">
            <h2>🚀 Démarrage rapide</h2>
            <ol>
                <li>Obtenez un token via <code>/api/v1/auth</code></li>
                <li>Utilisez le token dans les headers : <code>Authorization: Bearer YOUR_TOKEN</code></li>
                <li>Posez vos questions via <code>/api/v1/llm_chat</code></li>
            </ol>
        </div>
    </body>
    </html>
    """

# Inclure les routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")



