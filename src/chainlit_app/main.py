# src/chainlit_app/main.py
import chainlit as cl
import sys
import os

# Ajout du chemin racine
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.agents.master_agent_api import APIMasterAgent

# Instance globale de l'agent
master_agent = APIMasterAgent()

@cl.on_chat_start
async def start():
    """Initialisation de la session"""
    await cl.Message(
        content="""# 🏛️ Assistant Juridique Simple

Bonjour ! Je suis votre assistant spécialisé en droit du travail français.

Posez vos questions juridiques !""",
        author="Assistant"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Traitement simple sans validation humaine"""
    
    # Affichage du traitement
    async with cl.Step(name="🔍 Traitement en cours...") as step:
        try:
            # Génération de la réponse
            result = master_agent.query(message.content)
            
            if result["success"]:
                response = result["response"]
                confidence = result.get("confidence", "inconnue")
                
                step.output = f"Confiance: {confidence}"
                
                # Réponse directe
                await cl.Message(
                    content=f"""{response}

---

**Informations :**
- 📊 **Confiance :** {confidence}
- 🤖 **Traitement :** Automatique

⚠️ **Avertissement :** Réponse à titre informatif.""",
                    author="Assistant"
                ).send()
                
            else:
                step.output = "Erreur lors du traitement"
                await cl.Message(
                    content=f"❌ **Erreur :** {result.get('error', 'Erreur inconnue')}",
                    author="Assistant"
                ).send()
                
        except Exception as e:
            step.output = f"Erreur: {str(e)}"
            await cl.Message(
                content=f"❌ **Erreur technique :** {str(e)}",
                author="Assistant"
            ).send()

if __name__ == "__main__":
    cl.run()
