# src/api/llm_endpoint.py
from fastapi import Depends, APIRouter, HTTPException, status
from typing import Annotated
from src.api.schemas import AskQuestion
from src.api.models import User, Question, Answer, Conversation
from src.api.api_utils import get_current_user, db_dependency
from src.agents.master_agent_api import APIMasterAgent
from datetime import datetime

router = APIRouter(tags=["llm chat"])

@router.post("/llm_chat")
def llm_chat(
    user_data: AskQuestion, 
    current_user: Annotated[User, Depends(get_current_user)],
    db: db_dependency
):
    """Chat avec le Master Agent LLM"""
    
    try:
        # Validation
        if not user_data.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La question ne peut pas être vide"
            )
        
        # Création conversation
        conversation = Conversation(
            user_id=current_user.user_id,
            creation_date=datetime.now()
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        # Sauvegarde question
        question_record = Question(
            question=user_data.question,
            question_date=datetime.now(),
            user_id=current_user.user_id,
            conversation_id=conversation.conversation_id
        )
        db.add(question_record)
        db.commit()
        
        # ✅ Utilisation de l'agent API
        agent = APIMasterAgent()
        app = agent.build()
        
        initial_state = {
            "user_query": user_data.question,
            "droit_travail_response": "",
            "needs_cc": False,
            "cc_identified": None,
            "cc_response": "",
            "final_response": ""
        }
        
        thread_id = f"api_u{current_user.user_id}_c{conversation.conversation_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Exécution
        final_state = app.invoke(initial_state, config)
        response_text = final_state.get("final_response", "Erreur lors de la génération")
        
        # Sauvegarde réponse
        answer_record = Answer(
            answer=response_text,
            answer_date=datetime.now(),
            user_id=current_user.user_id,
            conversation_id=conversation.conversation_id
        )
        db.add(answer_record)
        db.commit()
        
        return {
            "success": True,
            "question": user_data.question,
            "answer": response_text,
            "conversation_id": conversation.conversation_id,
            "user": current_user.username,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne: {str(e)}"
        )
