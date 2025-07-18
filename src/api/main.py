from fastapi import FastAPI
from src.api.endpoints.auth_endpoint import router as auth_router
from src.api.endpoints.llm_chat_endpoint import router as llm_router



"""
Point d'entrée principal de l'API Master Agent LLM.

Ce module initialise l'application FastAPI et configure les routes
pour l'authentification et le chat avec le Master Agent.

Application Properties:
    title: "Master Agent LLM API"
    description: "API pour interactions avec le Master Agent spécialisé en droit du travail français"
    version: "1.0.0"

Routes incluses:
    - /api/v1/auth: Authentification et gestion des tokens
    - /api/v1/register: Création de comptes utilisateurs
    - /api/v1/llm_chat: Chat avec le Master Agent LLM

Note:
    L'API utilise FastAPI pour:
    - Documentation automatique (Swagger UI sur /docs)
    - Validation des données avec Pydantic
    - Gestion des routes avec APIRouter
    - Support asynchrone natif
"""


# Créer l'application FastAPI
app = FastAPI(
    title="Master Agent LLM API", 
    description="API pour interactions avec le Master Agent spécialisé en droit du travail français", 
    version="1.0.0"
)

# Inclure les routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
