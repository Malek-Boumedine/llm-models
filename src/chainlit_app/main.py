# src/chainlit_app/main.py
import chainlit as cl
import sys
import os

# Ajout du chemin racine
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.agents.droit_travail_agent import DroitTravailAgent
from src.chainlit_app.database import db_connection
from src.chainlit_app.auth import authenticate_user

# Instance globale de l'agent
droit_travail_agent = DroitTravailAgent()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Callback d'authentification Chainlit natif"""
    try:
        with next(db_connection()) as session:
            user = authenticate_user(session, username, password)
            if user:
                return cl.User(
                    identifier=username, 
                    metadata={"user_id": user.user_id, "role": user.role}
                )
    except Exception as e:
        print(f"Erreur auth: {e}")
    return None

@cl.on_chat_start
async def start():
    """Initialisation simple sans mémoire"""
    app_user = cl.user_session.get("user")
    
    await cl.Message(
        content=f"""# 🏛️ Assistant Juridique Français

👋 **Bienvenue {app_user.identifier} !**

Je suis votre assistant spécialisé en **droit du travail français et conventions collectives**.

- 📚 Code du travail complet
- 🏢 Conventions collectives étendues  
- 📰 Bulletins officiels (BOCC)

🆕 **Nouvelle session** - Chaque conversation est indépendante

💬 **Posez vos questions juridiques dès maintenant !**""",
        author="Assistant"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Traitement simple sans sauvegarde"""
    
    app_user = cl.user_session.get("user")
    
    # Appel direct de l'agent sans sauvegarde
    try:
        result = droit_travail_agent.query(message.content)
        
        if result.get("success"):
            response = result.get("response", "Erreur de réponse")
            
            
            # Affichage simple
            collections_used = result.get("collections_used", ["code_travail_collection", "conventions_etendues", "bocc"])
            
            await cl.Message(
                content=f"""{response}

---

**Informations :**
- 👤 **Utilisateur :** {app_user.identifier}
- 📚 **Collections :** {', '.join(collections_used)}

⚠️ **Avertissement :** Réponse à titre informatif. Consultez un professionnel pour des décisions importantes.""",
                author="Assistant"
            ).send()
            
        else:
            await cl.Message(
                content=f"❌ **Erreur :** {result.get('error', 'Erreur inconnue')}",
                author="Assistant"
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ **Erreur technique :** {str(e)}",
            author="Assistant"
        ).send()
        print(f"Erreur détaillée : {e}")

@cl.on_chat_end
async def end():
    """Fin de session simple"""
    app_user = cl.user_session.get("user")
    if app_user:
        await cl.Message(
            content=f"👋 **Session terminée** pour {app_user.identifier} - À bientôt !",
            author="Système"
        ).send()

