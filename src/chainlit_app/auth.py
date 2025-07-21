import bcrypt
from sqlmodel import Session, select
from src.chainlit_app.models import User
from typing import Optional





def hash_password(password: str) -> str:
    """Hache un mot de passe"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
    """Authentifie un utilisateur"""
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()
    
    if user and verify_password(password, user.hashed_password):
        return user
    return None

def create_user(session: Session, username: str, email: str, password: str) -> Optional[User]:
    """Crée un nouvel utilisateur"""
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = session.exec(select(User).where(User.username == username)).first()
        if existing_user:
            return None
            
        existing_email = session.exec(select(User).where(User.email == email)).first()
        if existing_email:
            return None
        
        # Créer le nouvel utilisateur
        hashed_pw = hash_password(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_pw
        )
        
        session.add(user)
        session.commit()
        session.refresh(user)
        
        return user
        
    except Exception as e:
        session.rollback()
        print(f"Erreur création utilisateur: {e}")
        return None
