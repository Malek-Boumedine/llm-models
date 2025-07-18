from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated
from sqlmodel import select
from src.api.schemas import CreateAccount
from src.api.api_utils import get_current_user, create_access_token, get_password_hash, authenticate_user, db_dependency
from src.api.models import User


router = APIRouter(tags=["auth"])


@router.post("/auth")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency):
    """Authentification utilisateur"""
    user = authenticate_user(identifiant=form_data.username, password=form_data.password, db=db)  # ✅ UNE SEULE LIGNE
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,  detail="Identifiant ou mot de passe incorrect.",  headers={"WWW-Authenticate": "Bearer"})
    
    token_data = {
        "sub": user.email,
        "extra": {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role
        }
    }
    access_token = create_access_token(data=token_data)
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "role": user.role
        }
    }


@router.post("/register")
def register(create_account_request: CreateAccount, db: db_dependency, current_user: Annotated[User, Depends(get_current_user)]):
    """Création d'un compte utilisateur (admin seulement)"""
    
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès refusé.")

    if create_account_request.password != create_account_request.password_confirm:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Les mots de passe ne correspondent pas.")

    # ✅ Vérification d'unicité
    existing_user = db.exec(
        select(User).where(
            (User.email == create_account_request.email) | 
            (User.username == create_account_request.username)
        )).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email ou nom d'utilisateur déjà utilisé."
        )

    user = User(
        username=create_account_request.username,
        email=create_account_request.email,
        hashed_password=get_password_hash(create_account_request.password),
        role=create_account_request.role
    )
    
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="Erreur lors de la création de l'utilisateur.")

    return {
        "message": f"Utilisateur {user.username} créé avec succès.",
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
    }    

