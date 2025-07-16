from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
from src.agents.web_agent import WebAgent
from src.agents.conventions_collectives_agent import ConventionsCollectivesAgent
from src.agents.droit_travail_agent import DroitTravailAgent
from langchain_ollama import ChatOllama
import requests
import os
from langgraph.checkpoint.memory import MemorySaver

# Structure de l'état
class MasterState(TypedDict):
    user_query: str
    droit_travail_response: str
    needs_cc: bool
    cc_identified: Optional[str]
    cc_response: str
    final_response: str
    analysis: str

class IntelligentMasterAgent:
    def __init__(self):
        self.web_agent = WebAgent()
        self.cc_agent = ConventionsCollectivesAgent()
        self.droit_agent = DroitTravailAgent()
        model_name = os.getenv("MODEL_NAME", "llama3.1:latest")
        self.llm = ChatOllama(model=model_name)
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    # =======================================================================
    # NODES
    # =======================================================================
    
    def fetch_general_labor_rules_node(self, state: MasterState):
        """Node obligatoire : appel à l'agent droit du travail"""
        print("🔍 Recherche des règles générales du droit du travail...")
        
        result = self.droit_agent.query(state["user_query"])
        
        return {
            "droit_travail_response": result.get("response", "Aucune réponse du droit du travail")
        }


    def analyze_cc_need_node(self, state: MasterState):
        """Node d'analyse : détermine si une convention collective est nécessaire"""
        print("🤖 Analyse de la nécessité d'une convention collective...")
        
        analysis_prompt = f"""
        Tu es un expert en droit du travail français et spécialiste des conventions collectives.

        CONTEXTE :
        Ton agent assistant "droit_agent" a déjà été consulté et a fourni une réponse générale basée sur le Code du travail.

        TA MISSION :
        Analyser si une convention collective spécifique est nécessaire pour compléter cette réponse et répondre avec précision à la question de l'utilisateur.

        QUESTION DE L'UTILISATEUR :
        {state['user_query']}

        RÉPONSE GÉNÉRALE DU DROIT DU TRAVAIL (déjà obtenue) :
        {state['droit_travail_response']}

        SCÉNARIOS POSSIBLES :

        📌 SI AUCUNE CONVENTION COLLECTIVE N'EST NÉCESSAIRE :
        - La réponse du droit du travail est complète et suffisante
        - La question porte sur des règles générales du Code du travail
        - Aucun secteur, entreprise ou profession spécifique n'est mentionné
        → Dans ce cas, seule la réponse du droit du travail sera utilisée

        📌 SI UNE CONVENTION COLLECTIVE EST NÉCESSAIRE :
        - La question fait référence à un secteur, entreprise, ou profession spécifique
        - La réponse générale mentionne des variations selon les conventions collectives
        - Le sujet concerne des éléments variables (salaires minima, classifications, primes, congés spécifiques, etc.)
        → Dans ce cas, ton agent assistant "cc_agent" sera consulté avec la convention collective identifiée

        CRITÈRES D'ANALYSE OBLIGATOIRES :
        1. ✓ Vérifier si la question fait référence explicitement ou implicitement à un secteur d'activité précis, une profession spécifique, une entreprise particulière ou un organisme précis
        2. ✓ Déterminer si la réponse générale mentionne des variations selon les conventions collectives
        3. ✓ Identifier si le sujet concerne des éléments traditionnellement régis par les conventions collectives (salaires minima, classification, durée de préavis, primes spécifiques, congés particuliers, temps de travail, indemnités conventionnelles, etc.)

        FORMAT OBLIGATOIRE POUR TA RÉPONSE :

        ANALYSE DE LA NÉCESSITÉ D'UNE CONVENTION COLLECTIVE :
        - RÉPONSE : **OUI** ou **NON**
        - JUSTIFICATION : [Explication concise des raisons précises motivant ta réponse, en citant les éléments pertinents de la question et/ou de la réponse générale]
        
        {f"- CONVENTION COLLECTIVE SUGGÉRÉE : [Si OUI, indique le titre exact ou le type de convention collective qui serait applicable]" if True else ""}

        EXEMPLES POUR T'AIDER :
        - "Quel est le SMIC en France ?" → **NON** (règle générale du Code du travail)
        - "Quel est le salaire minimum des employés de la CAF ?" → **OUI** (organisme spécifique + salaire conventionnel)
        - "Durée légale du travail ?" → **NON** (règle générale)
        - "Temps de travail des enseignants ?" → **OUI** (profession spécifique)
        - "Congés payés légaux ?" → **NON** (règle générale)
        - "Primes de fin d'année dans le commerce ?" → **OUI** (secteur spécifique + primes conventionnelles)

        INSTRUCTIONS FINALES :
        - Sois précis, rigoureux et factuel
        - N'invente aucune information
        - Si tu n'es pas certain, privilégie la prudence et suggère une vérification
        - Base-toi uniquement sur les éléments fournis dans la question et la réponse générale
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            analysis = response.content
            
            # Détection améliorée de la réponse
            needs_cc = "**OUI**" in analysis.upper() or "RÉPONSE : OUI" in analysis.upper()
            
            return {
                "needs_cc": needs_cc,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"Erreur d'analyse : {e}")
            return {
                "needs_cc": False,
                "analysis": f"Erreur d'analyse : {e}. Par précaution, aucune convention collective ne sera consultée."
            }
            

    def identify_cc_node(self, state: MasterState):
        """Node pour identifier précisément la convention collective applicable"""
        print("🔍 Identification précise de la convention collective applicable...")

        identification_prompt = f"""
        Tu es un expert en droit du travail français et spécialiste des conventions collectives.

        TA MISSION :
        Identifier précisément la convention collective applicable à la situation décrite dans la question posée par l'utilisateur.

        QUESTION DE L'UTILISATEUR :
        "{state['user_query']}"

        ÉTAPES À SUIVRE :
        1. Identifie clairement le secteur d'activité, la profession, l'organisme ou l'entreprise explicitement ou implicitement mentionné dans la question.
        2. Détermine précisément :
            - Le nom exact de la convention collective applicable
            - Le numéro IDCC correspondant à cette convention collective (format numérique à 5 chiffres)
            - Le secteur ou sous-secteur d'activité clairement concerné
        3. Si plusieurs conventions collectives peuvent correspondre, liste-les clairement en indiquant pour chacune :
            - Nom exact
            - Numéro IDCC précis
            - Secteur d'activité spécifique
        4. Si aucune convention collective n'est clairement identifiée, mentionne-le explicitement et conseille à l'utilisateur de préciser davantage ou de consulter Légifrance.

        FORMAT OBLIGATOIRE DE TA RÉPONSE :

        ✅ CONVENTION(S) COLLECTIVE(S) IDENTIFIÉE(S) :
        - Nom : [Nom exact]
        - IDCC : [Numéro IDCC à 5 chiffres]
        - Secteur d'activité : [Secteur précis]

        ⚠️ Si plusieurs conventions possibles :
        1. Nom : [Nom exact] - IDCC : [Numéro IDCC] - Secteur : [Secteur précis]
        2. Nom : [Nom exact] - IDCC : [Numéro IDCC] - Secteur : [Secteur précis]
        ...

        ❌ Si aucune convention clairement identifiée :
        "Aucune convention collective spécifique clairement identifiée. Merci de préciser davantage la question ou de consulter directement le site officiel Légifrance."

        Sois précis, rigoureux et factuel, n'extrapole jamais, et utilise exclusivement des informations fiables issues de ta base documentaire ou de sources officielles vérifiées.
        """

        # Premier essai avec l'agent web
        web_result = self.web_agent.query(identification_prompt)

        if web_result.get("success", False):
            cc_identified = web_result["response"]
        else:
            # Fallback : deuxième essai avec l'agent CC spécialisé
            cc_result = self.cc_agent.query(identification_prompt)
            cc_identified = cc_result.get("response", "Aucune convention collective spécifique clairement identifiée.")

        # Validation humaine avec possibilité de préciser
        print(f"\n📋 Convention collective identifiée :")
        print(cc_identified)

        human_validation = input("\nCette convention collective est-elle correcte ? (oui/non/skip): ")

        if human_validation.lower().strip() == "skip":
            return {
                "cc_identified": None,
                "needs_cc": False
            }
        elif human_validation.lower().strip() == "non":
            user_cc = input("Veuillez spécifier la convention collective correcte : ")
            return {
                "cc_identified": user_cc.strip() if user_cc.strip() else None
            }
        else:
            return {
                "cc_identified": cc_identified
            }


    def fetch_cc_rules_node(self, state: MasterState):
        """Node pour récupérer les règles spécifiques de la convention collective identifiée"""
        print("📄 Recherche des règles conventionnelles spécifiques...")

        cc_query = f"""
        Tu es un agent expert en conventions collectives françaises. 

        TA MISSION :
        Fournir une réponse précise et détaillée à la question posée par l'utilisateur, 
        en te basant exclusivement sur les règles spécifiques contenues dans la convention collective identifiée ci-dessous.

        QUESTION DE L'UTILISATEUR :
        {state['user_query']}

        CONVENTION COLLECTIVE APPLICABLE :
        {state['cc_identified']}

        RÈGLES À RESPECTER POUR TA RÉPONSE :
        1. Base-toi uniquement sur les dispositions précises de la convention collective spécifiée.
        2. Cite explicitement le numéro IDCC, le nom exact de la convention collective, et les articles pertinents (numéros, titres, sections) si disponibles.
        3. Détaille clairement les règles spécifiques prévues par la convention collective (exemples : salaires minima, durée du préavis, indemnités particulières, primes, temps de travail, congés spéciaux, classifications professionnelles, etc.).
        4. Si aucune règle spécifique n'est mentionnée clairement dans cette convention, indique-le explicitement et conseille de consulter directement le texte officiel sur Légifrance.

        FORMAT STRUCTURÉ OBLIGATOIRE POUR TA RÉPONSE :

        CONVENTION COLLECTIVE :
        - Nom exact : [Nom précis]
        - IDCC : [IDCC précis]

        RÉPONSE AUX RÈGLES SPÉCIFIQUES :
        - Articles pertinents : [Numéros et titres des articles]
        - Règles précises applicables : [Explications détaillées, critères, montants, durées, conditions, etc.]

        Si aucune règle spécifique trouvée :
        "Aucune règle spécifique clairement mentionnée dans cette convention collective concernant cette question. Consultez directement la convention collective sur Légifrance ou auprès d'un professionnel."

        Sois précis, rigoureux, clair et factuel dans ta réponse, et n'invente aucune information.
        """

        result = self.cc_agent.query(cc_query)

        return {
            "cc_response": result.get("response", "Aucune règle conventionnelle trouvée")
        }

    def compile_final_response_node(self, state: MasterState):
        """Node pour compiler la réponse finale"""
        print("📝 Compilation de la réponse finale...")
        
        if state["needs_cc"] and state.get("cc_identified"):
            # Réponse complète avec convention collective
            compiled_response = f"""
    📋 RÉPONSE COMPLÈTE

    ⚖️ RÈGLES GÉNÉRALES DU DROIT DU TRAVAIL :
    {state['droit_travail_response']}

    🏢 CONVENTION COLLECTIVE APPLICABLE :
    {state['cc_identified']}

    📄 RÈGLES CONVENTIONNELLES SPÉCIFIQUES :
    {state['cc_response']}
            """
        else:
            # Réponse basée uniquement sur le droit du travail
            compiled_response = f"""
    📋 RÉPONSE

    ⚖️ RÈGLES DU DROIT DU TRAVAIL :
    {state['droit_travail_response']}

    ℹ️ Cette réponse est basée sur les règles générales du droit du travail français.
            """
        
        return {
            "final_response": compiled_response
        }

    def send_final_response_node(self, state: MasterState):
        """Node final pour envoyer la réponse"""
        print("\n" + "="*60)
        print("✅ RÉPONSE FINALE")
        print("="*60)
        print(state["final_response"])
        print("="*60)
        
        return {}

    # =======================================================================
    # CONDITIONAL EDGES
    # =======================================================================
    
    def route_after_analysis(self, state: MasterState):
        """Route après l'analyse du besoin de convention collective"""
        if state["needs_cc"]:
            return "identify_cc"
        else:
            return "compile_final_response"

    def route_after_cc_identification(self, state: MasterState):
        """Route après l'identification de la convention collective"""
        if state.get("cc_identified") and state["needs_cc"]:
            return "fetch_cc_rules"
        else:
            return "compile_final_response"

    # =======================================================================
    # CONSTRUCTION DU WORKFLOW
    # =======================================================================
    
    def build(self):
        """Construit le workflow intelligent"""
        workflow = StateGraph(MasterState)
        
        # Ajout des nodes
        workflow.add_node("fetch_general_rules", self.fetch_general_labor_rules_node)
        workflow.add_node("analyze_cc_need", self.analyze_cc_need_node)
        workflow.add_node("identify_cc", self.identify_cc_node)
        workflow.add_node("fetch_cc_rules", self.fetch_cc_rules_node)
        workflow.add_node("compile_final_response", self.compile_final_response_node)
        workflow.add_node("send_final_response", self.send_final_response_node)
        
        # Définition du flux
        workflow.add_edge(START, "fetch_general_rules")
        workflow.add_edge("fetch_general_rules", "analyze_cc_need")
        
        workflow.add_conditional_edges(
            "analyze_cc_need",
            self.route_after_analysis,
            {
                "identify_cc": "identify_cc",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_conditional_edges(
            "identify_cc",
            self.route_after_cc_identification,
            {
                "fetch_cc_rules": "fetch_cc_rules",
                "compile_final_response": "compile_final_response"
            }
        )
        
        workflow.add_edge("fetch_cc_rules", "compile_final_response")
        workflow.add_edge("compile_final_response", "send_final_response")
        workflow.add_edge("send_final_response", END)
        
        # Compilation avec mémoire
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

# =======================================================================
# UTILISATION
# =======================================================================

def main():
    """Fonction principale"""
    print("🤖 Agent Master Intelligent - Questions Juridiques")
    print("=" * 60)
    
    agent = IntelligentMasterAgent()
    app = agent.build()
    
    user_query = input("\n❓ Votre question juridique : ")
    
    # État initial
    initial_state = {
        "user_query": user_query,
        "droit_travail_response": "",
        "needs_cc": False,
        "cc_identified": None,
        "cc_response": "",
        "final_response": "",
        "analysis": ""
    }
    
    # Configuration avec thread
    config = {"configurable": {"thread_id": "intelligent_session"}}
    
    # Exécution
    try:
        final_state = app.invoke(initial_state, config)
        print("\n✅ Session terminée avec succès")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")

if __name__ == "__main__":
    main()
