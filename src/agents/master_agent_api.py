from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from src.agents.web_agent import WebAgent
from src.agents.conventions_collectives_agent import ConventionsCollectivesAgent
from src.agents.droit_travail_agent import DroitTravailAgent
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity
import os
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Structure de l'état améliorée
class MasterState(TypedDict):
    user_query: str
    droit_travail_response: str
    droit_travail_confidence: str
    needs_cc: bool
    cc_analysis: str
    cc_identified: Optional[str]
    cc_response: str
    cc_confidence: str
    web_fallback_used: bool
    web_response: str
    final_response: str
    confidence_level: str

class APIMasterAgent:
    """Master Agent API optimisé avec prompts améliorés"""
    
    def __init__(self, model_type=None):
        # Configuration du modèle
        self.model_type = model_type or os.getenv("MODEL_TYPE", "local")
        self.llm = self._initialize_llm()
        
        # Agents spécialisés
        self.web_agent = WebAgent()
        self.cc_agent = ConventionsCollectivesAgent()
        self.droit_agent = DroitTravailAgent()

    def _initialize_llm(self):
        """Initialise le LLM selon le type configuré"""
        if self.model_type == "local":
            return ChatOllama(model=os.getenv("MODEL_NAME", "llama3.1:latest"))
        elif self.model_type == "groq":
            return ChatGroq(
                model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2
            )
        elif self.model_type == "perplexity":
            return ChatPerplexity(
                model=os.getenv("PERPLEXITY_MODEL", "sonar"),
                pplx_api_key=os.getenv("PERPLEXITY_API_KEY"),
                temperature=0.2
            )

    # =======================================================================
    # NODES OPTIMISÉS
    # =======================================================================
    
    def fetch_general_labor_rules_node(self, state: MasterState):
        """Récupère les règles générales du droit du travail"""
        try:
            result = self.droit_agent.query(state["user_query"])
            response = result.get("response", "Aucune réponse du droit du travail")
            
            # Extraction du niveau de confiance de la réponse
            confidence = self._extract_confidence_from_response(response)
            
            return {
                "droit_travail_response": response,
                "droit_travail_confidence": confidence
            }
        except Exception as e:
            return {
                "droit_travail_response": f"Erreur lors de la consultation du droit du travail : {str(e)}",
                "droit_travail_confidence": "basse"
            }

    def analyze_cc_need_node(self, state: MasterState):
        """Analyse si une convention collective est nécessaire avec prompt optimisé"""
        
        analysis_prompt = f"""
        Tu es un expert juridique qui détermine si une convention collective est nécessaire.

        **DONNÉES À ANALYSER :**
        - Question : {state['user_query']}
        - Réponse droit du travail : {state['droit_travail_response']}
        - Confiance : {state['droit_travail_confidence']}

        **ALGORITHME DE DÉCISION (par ordre de priorité) :**

        **1. CRITÈRES PRIORITAIRES → Convention collective NÉCESSAIRE :**
        - ✅ Mention explicite de secteur/profession (boulangerie, BTP, banques, etc.)
        - ✅ Question sur salaires minimums, primes, classifications
        - ✅ Réponse mentionne "selon convention collective" ou "IDCC"
        - ✅ Temps de travail spécifique (horaires décalés, nuit, etc.)
        - ✅ Avantages sociaux sectoriels (mutuelle, prévoyance, etc.)

        **2. CRITÈRES SECONDAIRES → Évaluation contextuelle :**
        - ⚠️ Confiance droit du travail < 0.8
        - ⚠️ Entreprise/employeur spécifique mentionné
        - ⚠️ Réponse générale mais secteur implicite détectable

        **3. CRITÈRES D'EXCLUSION → Convention collective NON NÉCESSAIRE :**
        - ❌ SMIC national, durée légale 35h
        - ❌ Procédures générales (licenciement, démission)
        - ❌ Congés légaux minimums (5 semaines)
        - ❌ Règles sécurité générales
        - ❌ Réponse complète ET confiance > 0.8

        **INSTRUCTIONS D'ANALYSE :**
        1. Vérifie d'abord les critères prioritaires (ordre d'importance)
        2. Si aucun critère prioritaire → évalue les critères secondaires
        3. Si critères d'exclusion → réponds NON directement
        4. En cas de doute → privilégie OUI (principe de précaution)

        **FORMAT DE RÉPONSE OBLIGATOIRE :**
        "OUI" ou "NON" + justification en 1 phrase précise avec le critère utilisé.

        **EXEMPLES :**
        - "OUI - Secteur boulangerie mentionné (critère prioritaire)"
        - "NON - Question générale sur congés légaux (critère d'exclusion)"
        - "OUI - Confiance faible 0.6 nécessite vérification sectorielle"
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            analysis = response.content.strip()
            needs_cc = "OUI" in analysis.upper()[:10]  # Vérifie dans les 10 premiers caractères
            
            return {
                "needs_cc": needs_cc,
                "cc_analysis": analysis
            }
        except Exception as e:
            return {
                "needs_cc": False,
                "cc_analysis": f"Erreur d'analyse : {str(e)}"
            }

    def identify_cc_node(self, state: MasterState):
        """Identifie la convention collective avec prompt optimisé"""
        
        identification_prompt = f"""
        Identifier la convention collective applicable pour cette question :

        QUESTION : {state['user_query']}
        ANALYSE : {state['cc_analysis']}

        **Méthode d'identification :**
        1. Extraire les éléments clés : secteur, entreprise, profession
        2. Identifier l'IDCC correspondant
        3. Vérifier que la convention est étendue

        **Éléments à rechercher :**
        - Nom exact de l'entreprise ou secteur
        - Code APE si mentionné
        - Profession spécifique
        - Effectifs de l'entreprise
        - Activité principale

        **Conventions courantes :**
        - CAF/CPAM/URSSAF → IDCC 218 (Organismes de sécurité sociale)
        - Commerce → IDCC 3305 (Commerce de détail)
        - BTP → IDCC 1597 (Bâtiment)
        - Banques → IDCC 2120 (Banques)
        - Métallurgie → IDCC 3248 (Métallurgie)

        **Format de réponse strict :**
        IDCC : [Numéro]
        Nom : [Nom complet de la convention]
        Secteur : [Secteur d'activité]
        Pertinence : [Pourquoi cette convention s'applique]

        Si aucune convention identifiable :
        "CONVENTION NON IDENTIFIABLE - Éléments manquants : [liste des éléments nécessaires]"
        """

        try:
            # Utilisation de l'agent web pour identifier la convention
            web_result = self.web_agent.query(identification_prompt)
            
            if web_result.get("success", False) and self._is_valid_cc_response(web_result["response"]):
                cc_identified = web_result["response"]
            else:
                # Fallback avec une recherche directe
                cc_identified = self._fallback_cc_identification(state["user_query"])

            # Validation automatique
            if self._auto_validate_cc(cc_identified):
                return {
                    "cc_identified": cc_identified
                }
            else:
                return {
                    "cc_identified": None,
                    "needs_cc": False
                }
                
        except Exception as e:
            return {
                "cc_identified": None,
                "needs_cc": False
            }

    def fetch_cc_rules_node(self, state: MasterState):
        """Récupère les règles de la convention collective avec prompt optimisé"""
        
        cc_query = f"""
        Rechercher les règles spécifiques de cette convention collective :

        QUESTION INITIALE : {state['user_query']}
        CONVENTION IDENTIFIÉE : {state['cc_identified']}
        
        **Instructions spécifiques :**
        - Utilise le format de réponse standardisé avec IDCC, articles précis
        - Vérifie les métadonnées pour confirmer l'IDCC
        - Indique le niveau de certitude
        - Précise si la règle est plus favorable que le Code du travail
        
        Applique ton prompt système pour la précision juridique.
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
                "cc_response": f"Erreur lors de la consultation des conventions : {str(e)}",
                "cc_confidence": "basse"
            }

    def web_fallback_node(self, state: MasterState):
        """Utilise l'agent web comme fallback si confiance faible"""
        
        web_query = f"""
        Rechercher des informations officielles récentes sur :

        QUESTION : {state['user_query']}
        
        Priorité aux sources gouvernementales françaises (.gouv.fr, Légifrance).
        Cherche des actualités, réformes récentes ou précisions officielles.
        """
        
        try:
            result = self.web_agent.query(web_query)
            return {
                "web_fallback_used": True,
                "web_response": result.get("response", "Aucune information web trouvée")
            }
        except Exception as e:
            return {
                "web_fallback_used": True,
                "web_response": f"Erreur de recherche web : {str(e)}"
            }

    def compile_final_response_node(self, state: MasterState):
        """Compile la réponse finale avec prompt de synthèse optimisé"""
        
        compilation_prompt = f"""
        Compiler une réponse juridique complète et fiable :

        QUESTION : {state['user_query']}
        DROIT DU TRAVAIL : {state['droit_travail_response']}
        CONFIANCE DT : {state['droit_travail_confidence']}
        CONVENTION COLLECTIVE : {state.get('cc_response', 'Non applicable')}
        CONFIANCE CC : {state.get('cc_confidence', 'Non applicable')}
        RECHERCHE WEB : {state.get('web_response', 'Non utilisée')}

        **Objectif :** Synthèse claire et hiérarchisée

        **Structure obligatoire :**
        📋 **RÉPONSE JURIDIQUE COMPLÈTE**

        ⚖️ **RÈGLE GÉNÉRALE (Code du travail) :**
        [Synthèse claire de la réponse droit du travail]

        🏢 **RÈGLE SPÉCIFIQUE (Convention collective) :**
        [Synthèse de la réponse convention collective SI applicable]

        🌐 **INFORMATIONS COMPLÉMENTAIRES :**
        [Informations web SI utilisées]

        🎯 **RÈGLE APPLICABLE :**
        [Quelle règle s'applique en priorité et pourquoi]

        📊 **NIVEAU DE CONFIANCE :** [HAUTE/MOYENNE/BASSE]

        ⚠️ **RECOMMANDATIONS :**
        - Vérifier l'IDCC de votre entreprise
        - Consulter votre convention collective complète
        - En cas de doute, consulter un professionnel du droit

        **Instructions :**
        - Hiérarchise les règles (convention > loi si plus favorable)
        - Indique clairement quelle règle s'applique
        - Signale les cas d'incertitude
        - Reste factuel et précis
        - Évalue la confiance globale
        """
        
        try:
            response = self.llm.invoke(compilation_prompt)
            compiled_response = response.content
            
            # Évaluation de la confiance globale
            overall_confidence = self._evaluate_overall_confidence(state)
            
            return {
                "final_response": compiled_response,
                "confidence_level": overall_confidence
            }
        except Exception as e:
            # Fallback : compilation simple
            fallback_response = self._create_fallback_response(state)
            return {
                "final_response": fallback_response,
                "confidence_level": "basse"
            }

    # =======================================================================
    # MÉTHODES UTILITAIRES AMÉLIORÉES
    # =======================================================================
    
    def _extract_confidence_from_response(self, response: str) -> str:
        """Extrait le niveau de confiance d'une réponse"""
        response_lower = response.lower()
        
        if "certitude" in response_lower:
            if "haute" in response_lower:
                return "haute"
            elif "moyenne" in response_lower:
                return "moyenne"
            elif "basse" in response_lower:
                return "basse"
        
        # Analyse par indicateurs
        confidence_indicators = ["idcc", "article", "précis", "exact"]
        uncertainty_indicators = ["incertain", "possiblement", "peut-être"]
        
        has_confidence = any(indicator in response_lower for indicator in confidence_indicators)
        has_uncertainty = any(indicator in response_lower for indicator in uncertainty_indicators)
        
        if has_confidence and not has_uncertainty:
            return "haute"
        elif has_confidence or not has_uncertainty:
            return "moyenne"
        else:
            return "basse"
    
    def _is_valid_cc_response(self, response: str) -> bool:
        """Valide si la réponse contient une convention collective valide"""
        indicators = ["IDCC", "convention collective", "nom :", "secteur :"]
        return any(indicator.lower() in response.lower() for indicator in indicators)
    
    def _auto_validate_cc(self, cc_response: str) -> bool:
        """Validation automatique améliorée de la convention collective"""
        if not cc_response or len(cc_response.strip()) < 20:
            return False
        
        # Vérifications positives
        positive_indicators = [
            "idcc" in cc_response.lower(),
            "convention" in cc_response.lower(),
            any(f"idcc {num}" in cc_response.lower() for num in ["218", "3305", "1597", "2120", "3248"])
        ]
        
        # Vérifications négatives
        negative_indicators = [
            "non identifiable" in cc_response.lower(),
            "aucune convention" in cc_response.lower(),
            "impossible" in cc_response.lower(),
            "éléments manquants" in cc_response.lower()
        ]
        
        return any(positive_indicators) and not any(negative_indicators)
    
    def _fallback_cc_identification(self, query: str) -> str:
        """Identification de fallback basée sur mots-clés"""
        query_lower = query.lower()
        
        # Mapping simplifié
        cc_mapping = {
            "caf": "IDCC 218 - Convention collective des organismes de sécurité sociale",
            "cpam": "IDCC 218 - Convention collective des organismes de sécurité sociale",
            "urssaf": "IDCC 218 - Convention collective des organismes de sécurité sociale",
            "commerce": "IDCC 3305 - Convention collective du commerce de détail",
            "btp": "IDCC 1597 - Convention collective du bâtiment",
            "banque": "IDCC 2120 - Convention collective des banques",
            "métallurgie": "IDCC 3248 - Convention collective de la métallurgie"
        }
        
        for keyword, cc in cc_mapping.items():
            if keyword in query_lower:
                return cc
        
        return "CONVENTION NON IDENTIFIABLE - Secteur non spécifié"
    
    def _evaluate_overall_confidence(self, state: MasterState) -> str:
        """Évalue la confiance globale de la réponse"""
        dt_conf = state.get("droit_travail_confidence", "basse")
        cc_conf = state.get("cc_confidence", "non applicable")
        
        if dt_conf == "haute" and (cc_conf == "haute" or cc_conf == "non applicable"):
            return "haute"
        elif dt_conf in ["haute", "moyenne"] and cc_conf in ["haute", "moyenne", "non applicable"]:
            return "moyenne"
        else:
            return "basse"
    
    def _create_fallback_response(self, state: MasterState) -> str:
        """Crée une réponse de fallback en cas d'erreur"""
        return f"""📋 **RÉPONSE JURIDIQUE**

⚖️ **RÈGLE GÉNÉRALE :**
{state['droit_travail_response']}

⚠️ **AVERTISSEMENT :** Réponse compilée automatiquement. Pour des décisions juridiques importantes, consultez un professionnel du droit.
"""

    # =======================================================================
    # ROUTING LOGIC AMÉLIORÉ
    # =======================================================================
    
    def route_after_analysis(self, state: MasterState) -> str:
        """Route après analyse avec gestion de la confiance"""
        if state["needs_cc"]:
            return "identify_cc"
        elif state["droit_travail_confidence"] == "basse":
            return "web_fallback"
        else:
            return "compile_final_response"

    def route_after_cc_identification(self, state: MasterState) -> str:
        """Route après identification CC"""
        if state.get("cc_identified") and state["needs_cc"]:
            return "fetch_cc_rules"
        elif state["droit_travail_confidence"] == "basse":
            return "web_fallback"
        else:
            return "compile_final_response"
    
    def route_after_cc_rules(self, state: MasterState) -> str:
        """Route après récupération des règles CC"""
        if (state["droit_travail_confidence"] == "basse" and 
            state.get("cc_confidence", "basse") == "basse"):
            return "web_fallback"
        else:
            return "compile_final_response"

    # =======================================================================
    # CONSTRUCTION DU WORKFLOW OPTIMISÉ
    # =======================================================================
    
    def build(self):
        """Construit le workflow API optimisé"""
        workflow = StateGraph(MasterState)
        
        # Ajout des nodes
        workflow.add_node("fetch_general_rules", self.fetch_general_labor_rules_node)
        workflow.add_node("analyze_cc_need", self.analyze_cc_need_node)
        workflow.add_node("identify_cc", self.identify_cc_node)
        workflow.add_node("fetch_cc_rules", self.fetch_cc_rules_node)
        workflow.add_node("web_fallback", self.web_fallback_node)
        workflow.add_node("compile_final_response", self.compile_final_response_node)
        
        # Définition du flux optimisé
        workflow.add_edge(START, "fetch_general_rules")
        workflow.add_edge("fetch_general_rules", "analyze_cc_need")
        
        workflow.add_conditional_edges(
            "analyze_cc_need",
            self.route_after_analysis,
            {
                "identify_cc": "identify_cc",
                "web_fallback": "web_fallback",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_conditional_edges(
            "identify_cc",
            self.route_after_cc_identification,
            {
                "fetch_cc_rules": "fetch_cc_rules",
                "web_fallback": "web_fallback",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_conditional_edges(
            "fetch_cc_rules",
            self.route_after_cc_rules,
            {
                "web_fallback": "web_fallback",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_edge("web_fallback", "compile_final_response")
        workflow.add_edge("compile_final_response", END)
        
        # Compilation avec mémoire
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    # =======================================================================
    # MÉTHODE D'INTERFACE SIMPLIFIÉE
    # =======================================================================
    
    def query(self, user_query: str, thread_id: str = "default") -> dict:
        """Interface simplifiée pour l'API"""
        app = self.build()
        
        initial_state = {
            "user_query": user_query,
            "droit_travail_response": "",
            "droit_travail_confidence": "",
            "needs_cc": False,
            "cc_analysis": "",
            "cc_identified": None,
            "cc_response": "",
            "cc_confidence": "",
            "web_fallback_used": False,
            "web_response": "",
            "final_response": "",
            "confidence_level": ""
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = app.invoke(initial_state, config)
            
            return {
                "success": True,
                "response": final_state.get("final_response", "Erreur lors de la génération"),
                "confidence": final_state.get("confidence_level", "basse"),
                "cc_used": bool(final_state.get("cc_identified")),
                "web_used": final_state.get("web_fallback_used", False),
                "thread_id": thread_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Erreur lors du traitement de la demande",
                "confidence": "basse"
            }

# =======================================================================
# TEST AMÉLIORÉ
# =======================================================================

def test_optimized_agent():
    """Test de l'agent optimisé"""
    agent = APIMasterAgent()
    
    test_cases = [
        "Quel est le SMIC en France ?",
        "Quelle est la prime d'ancienneté à la CAF ?",
        "Nouveautés du code du travail en 2025"
    ]
    
    for question in test_cases:
        print(f"\n🔍 Test : {question}")
        result = agent.query(question)
        
        if result["success"]:
            print(f"✅ Succès - Confiance: {result['confidence']}")
            print(f"📄 Réponse: {result['response'][:200]}...")
        else:
            print(f"❌ Erreur: {result['error']}")

if __name__ == "__main__":
    test_optimized_agent()
