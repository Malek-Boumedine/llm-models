from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from src.agents.conventions_collectives_agent import ConventionsCollectivesAgent
from src.agents.droit_travail_agent import DroitTravailAgent
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity
import os
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import warnings


warnings.filterwarnings("ignore")


load_dotenv()

# Structure de l'état simplifiée
class MasterState(TypedDict):
    user_query: str
    droit_travail_response: str
    droit_travail_confidence: str
    needs_cc: bool
    cc_analysis: str
    cc_identified: Optional[str]
    cc_response: str
    cc_confidence: str
    final_response: str
    confidence_level: str

class APIMasterAgent:
    """Master Agent simplifié utilisant uniquement les 2 agents spécialisés"""
    
    def __init__(self, model_type=None):
        # Configuration du modèle
        self.model_type = model_type or os.getenv("MODEL_TYPE", "groq")  # Groq par défaut
        self.llm = self._initialize_llm()
        
        # Agents spécialisés (seulement 2)
        self.cc_agent = ConventionsCollectivesAgent()
        self.droit_agent = DroitTravailAgent()

    def _initialize_llm(self):
        """Initialise le LLM selon le type configuré"""
        if self.model_type == "local":
            return ChatOllama(
                model=os.getenv("MODEL_NAME", "llama3.1:latest"),
                temperature=0
            )
        elif self.model_type == "groq":
            return ChatGroq(
                model=os.getenv("GROQ_MODEL", "meta-llama/llama-3.1-70b-instruct"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0
            )
        elif self.model_type == "perplexity":
            return ChatPerplexity(
                model=os.getenv("PERPLEXITY_MODEL", "sonar"),
                pplx_api_key=os.getenv("PERPLEXITY_API_KEY"),
                temperature=0
            )

    # =======================================================================
    # NODES SIMPLIFIÉS
    # =======================================================================
    
    def fetch_general_labor_rules_node(self, state: MasterState):
        """Récupère les règles générales du droit du travail"""
        try:
            result = self.droit_agent.query(state["user_query"])
            response = result.get("response", "Aucune réponse du droit du travail")
            confidence = self._extract_confidence_from_response(response)
            
            return {
                "droit_travail_response": response,
                "droit_travail_confidence": confidence
            }
        except Exception as e:
            return {
                "droit_travail_response": f"Erreur : {str(e)}",
                "droit_travail_confidence": "basse"
            }

    def analyze_cc_need_node(self, state: MasterState):
        """Analyse si une convention collective est nécessaire"""
        
        analysis_prompt = f"""Tu es un expert juridique français de niveau maître. Ta mission est de fournir des réponses d'excellence absolue.

        QUESTION: {state['user_query']}
        CONTEXTE DISPONIBLE: {state['droit_travail_response']}
        CONVENTION COLLECTIVE: {state.get('cc_response', 'Non consultée')}

        STANDARDS D'EXCELLENCE OBLIGATOIRES:

        1. PRÉCISION FACTUELLE ABSOLUE
        - Chiffres exacts (€, heures, %, jours, taux)
        - Calculs corrects (SMIC mensuel = horaire × 151,67h, pas × 4 semaines)
        - Dates précises des textes et réformes
        - Références légales complètes (articles, décrets, numéros)

        2. CALCULS ET FORMULES EXACTES
        - Durées: heures réglementaires, majorations, repos
        - Montants: salaires, indemnités, primes (brut/net si pertinent)
        - Taux: majoration heures sup (25%, 50%), congés payés (10%)
        - Délais: préavis, prescription, procédures

        3. HIÉRARCHIE JURIDIQUE RESPECTÉE
        - Code du travail (base légale minimum)
        - Convention collective (si plus favorable)
        - Accord d'entreprise (si plus favorable encore)
        - Jurisprudence récente si pertinente

        4. STRUCTURE DE RÉPONSE PROFESSIONNELLE
        - Réponse directe en première ligne
        - Montants/durées/taux en chiffres clairs
        - Base légale précise (article + référence)
        - Conditions d'application si nécessaires
        - Exceptions ou cas particuliers

        5. GESTION DE L'INCERTITUDE
        - Si données manquantes: "Information non disponible dans mes sources"
        - Si estimation: "Estimation basée sur [source], à vérifier officiellement"
        - Si évolution récente: "Sous réserve des dernières modifications"

        EXEMPLE D'EXCELLENCE:
        Question SMIC → "11,27€/heure brut depuis le 1er janvier 2025. Mensuel: 1 709,85€ brut (151,67h). Base: Art. L.3231-1 Code du travail, décret n°2024-963."

        Question Heures sup → "Majoration 25% pour heures 36-43, 50% au-delà. Base: Art. L.3121-22 Code du travail. Calcul: salaire horaire × 1,25 ou 1,50."

        CONSIGNE FINALE: Réponds avec l'excellence d'un consultant juridique senior. Précision absolue, structure claire, références exactes.

        RÉPONSE:"""
        
        try:
            response = self.llm.invoke(analysis_prompt)
            analysis = response.content.strip()
            needs_cc = "OUI" in analysis.upper()[:10]
            
            return {
                "needs_cc": needs_cc,
                "cc_analysis": analysis
            }
        except Exception as e:
            return {
                "needs_cc": False,
                "cc_analysis": f"Erreur d'analyse : {str(e)}"
            }

    def fetch_cc_rules_node(self, state: MasterState):
        """Récupère les règles de la convention collective"""
        
        # Construction d'une query optimisée pour l'agent CC
        cc_query = f"""
        QUESTION ORIGINALE : {state['user_query']}
        ANALYSE : {state['cc_analysis']}
        
        Recherche les règles spécifiques des conventions collectives pour cette question.
        Utilise ton format de réponse standardisé avec ÉTAPES.
        """
        
        try:
            result = self.cc_agent.query(cc_query)
            response = result.get("response", "Aucune règle conventionnelle trouvée")
            confidence = self._extract_confidence_from_response(response)
            
            return {
                "cc_response": response,
                "cc_confidence": confidence
            }
        except Exception as e:
            return {
                "cc_response": f"Erreur conventions : {str(e)}",
                "cc_confidence": "basse"
            }

    def compile_final_response_node(self, state: MasterState):
        """Compile la réponse finale"""
        
        compilation_prompt = f"""Tu es un assistant juridique français. Réponds de façon directe et factuelle.

        QUESTION: {state['user_query']}
        CONTEXTE: {state['droit_travail_response']}

        RÈGLES:
        - Donne des chiffres précis (montants, durées, taux)
        - Si tu ne connais pas exactement, donne ta meilleure estimation
        - Sois direct, pas de formules creuses
        - Structure simple: Réponse + Base légale + Montant/Durée si applicable

        EXEMPLE SMIC: "Le SMIC horaire est de 11,88€ brut (estimation 2025). Base: Articles L.3231-1 Code du travail."

        Réponds maintenant:"""
        
        try:
            response = self.llm.invoke(compilation_prompt)
            compiled_response = response.content
            overall_confidence = self._evaluate_overall_confidence(state)
            
            return {
                "final_response": compiled_response,
                "confidence_level": overall_confidence
            }
        except Exception as e:
            # Réponse de fallback simple
            fallback = f"""🏛️ **RÉPONSE JURIDIQUE**

⚖️ **RÈGLE GÉNÉRALE :**
{state['droit_travail_response']}

⚠️ **Avertissement :** Erreur lors de la compilation. Consultez un professionnel du droit."""
            
            return {
                "final_response": fallback,
                "confidence_level": "basse"
            }

    # =======================================================================
    # MÉTHODES UTILITAIRES
    # =======================================================================
    
    def _extract_confidence_from_response(self, response: str) -> str:
        """Extrait le niveau de confiance d'une réponse"""
        if not response or response is None:
            return "basse"
        response_lower = response.lower()
        
        # Recherche directe de "certitude"
        if "certitude" in response_lower:
            if "haute" in response_lower:
                return "haute"
            elif "moyenne" in response_lower:
                return "moyenne"
            elif "basse" in response_lower:
                return "basse"
        
        # Analyse par indicateurs
        confidence_indicators = ["idcc", "article", "précis", "trouvé"]
        uncertainty_indicators = ["non trouvé", "incertain", "peut-être", "aucune"]
        
        has_confidence = any(ind in response_lower for ind in confidence_indicators)
        has_uncertainty = any(ind in response_lower for ind in uncertainty_indicators)
        
        if has_confidence and not has_uncertainty:
            return "haute"
        elif has_confidence or not has_uncertainty:
            return "moyenne"
        else:
            return "basse"
    
    def _evaluate_overall_confidence(self, state: MasterState) -> str:
        """Évalue la confiance globale de la réponse"""
        dt_conf = state.get("droit_travail_confidence", "basse")
        cc_conf = state.get("cc_confidence", "non applicable")
        
        # Logique simplifiée
        if dt_conf == "haute" and (cc_conf == "haute" or not state["needs_cc"]):
            return "haute"
        elif dt_conf in ["haute", "moyenne"] and cc_conf in ["haute", "moyenne", "non applicable"]:
            return "moyenne"
        else:
            return "basse"

    # =======================================================================
    # ROUTING SIMPLIFIÉ
    # =======================================================================
    
    def route_after_analysis(self, state: MasterState) -> str:
        """Route après analyse du besoin de convention collective"""
        return "fetch_cc_rules" if state["needs_cc"] else "compile_final_response"

    # =======================================================================
    # CONSTRUCTION DU WORKFLOW SIMPLIFIÉ
    # =======================================================================
    
    def build(self):
        """Construit le workflow simplifié"""
        workflow = StateGraph(MasterState)
        
        # Nodes simplifiés (seulement 4)
        workflow.add_node("fetch_general_rules", self.fetch_general_labor_rules_node)
        workflow.add_node("analyze_cc_need", self.analyze_cc_need_node)
        workflow.add_node("fetch_cc_rules", self.fetch_cc_rules_node)
        workflow.add_node("compile_final_response", self.compile_final_response_node)
        
        # Flux linéaire simplifié
        workflow.add_edge(START, "fetch_general_rules")
        workflow.add_edge("fetch_general_rules", "analyze_cc_need")
        
        # Route conditionnelle unique
        workflow.add_conditional_edges(
            "analyze_cc_need",
            self.route_after_analysis,
            {
                "fetch_cc_rules": "fetch_cc_rules",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_edge("fetch_cc_rules", "compile_final_response")
        workflow.add_edge("compile_final_response", END)
        
        # Compilation
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    # =======================================================================
    # INTERFACE API SIMPLIFIÉE
    # =======================================================================
    
    def query(self, user_query: str, conversation_history: list = None, thread_id: str = "default") -> dict:
        """Interface simplifiée pour l'API avec mémoire conversationnelle"""
        app = self.build()
        
        # 🧠 INTÉGRATION MÉMOIRE : Formatage contexte conversationnel
        enhanced_query = user_query
        if conversation_history and len(conversation_history) > 0:
            # Prendre les 5 derniers échanges pour le contexte
            recent_history = conversation_history[-5:]
            
            context = "\n--- CONTEXTE CONVERSATIONNEL PRÉCÉDENT ---\n"
            for msg in recent_history:
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
                context += f"{role}: {content}\n"
            context += "--- FIN CONTEXTE ---\n\nQUESTION ACTUELLE: "
            
            enhanced_query = context + user_query
        
        initial_state = {
            "user_query": enhanced_query,
            "droit_travail_response": "",
            "droit_travail_confidence": "",
            "needs_cc": False,
            "cc_analysis": "",
            "cc_identified": None,
            "cc_response": "",
            "cc_confidence": "",
            "final_response": "",
            "confidence_level": ""
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = app.invoke(initial_state, config)
            
            return {
                "success": True,
                "response": final_state.get("final_response", "Erreur"),
                # 🔥 CONFIDENCE SUPPRIMÉ COMPLÈTEMENT
                "cc_used": bool(final_state.get("cc_response") and 
                            "non consultée" not in final_state.get("cc_response", "")),
                "thread_id": thread_id,
                "memory_used": len(conversation_history) if conversation_history else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Erreur lors du traitement"
            }

# =======================================================================
# TEST SIMPLIFIÉ
# =======================================================================

def test_simple_agent():
    """Test de l'agent simplifié"""
    agent = APIMasterAgent()
    
    test_questions = [
        "Quel est le montant exact en euros du SMIC en France depuis janvier 2025 ?",
        "Quelle est la majoration des heures de nuit en boulangerie ?",
        "Durée du préavis de démission ?"
    ]
    
    for question in test_questions:
        print(f"\n📋 Test : {question}")
        result = agent.query(question)
        
        if result["success"]:
            print(f"✅ Confiance: {result['confidence']}")
            print(f"🏢 CC utilisée: {result['cc_used']}")
            print(f"📄 Réponse: {result['response'][:300]}...")
        else:
            print(f"❌ Erreur: {result['error']}")

if __name__ == "__main__":
    test_simple_agent()
