from sqlmodel import Session, select
from src.chainlit_app.models import User
from typing import Optional

def get_user_by_username(session: Session, username: str) -> Optional[User]:
    """Récupère un utilisateur par son nom d'utilisateur"""
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()

def create_simple_user(session: Session, username: str, password_hash: str, role: str = "user") -> User:
    """Crée un utilisateur simple pour le POC"""
    try:
        new_user = User(
            username=username,
            password_hash=password_hash,
            role=role
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return new_user
    except Exception as e:
        session.rollback()
        print(f"Erreur création utilisateur: {e}")
        raise e

# Toutes les autres fonctions de conversation supprimées pour le POC
