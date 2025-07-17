# gestion des tokens
from datetime import datetime, timedelta, timezone
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import os
from dotenv import load_dotenv
from passlib.context import CryptContext
from src.api.database import Session, db_connection
from jose import JWTError, jwt
from src.api.models import User
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import select



router = APIRouter(tags=["auth"])   # pour les routes d'authentification

load_dotenv(dotenv_path="./app/.env") 
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 15))

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="login") 


db_dependency = Annotated[Session, Depends(db_connection)]   # dépendance pour la connexion à la BDD

def create_access_token(data: dict, expires_delta : timedelta = None) -> str:
    """
    Crée un token JWT d'accès.

    Args:
        data (dict): Données à encoder dans le token
        expires_delta (timedelta, optional): Durée de validité personnalisée

    Returns:
        str: Token JWT encodé

    Note:
        - Utilise la SECRET_KEY et l'ALGORITHM définis
        - Durée par défaut définie par ACCESS_TOKEN_EXPIRE_MINUTES
        - Inclut automatiquement la date d'expiration
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_password_hash(password : str) -> str: 
    """
    Génère un hash bcrypt d'un mot de passe.

    Args:
        password (str): Mot de passe en clair

    Returns:
        str: Hash bcrypt du mot de passe

    Note:
        Utilise l'algorithme bcrypt avec salt automatique
    """
    return bcrypt_context.hash(password)


def authenticate_user(email : str, password : str, db : db_dependency) -> User : 
    """
    Authentifie un utilisateur par email et mot de passe.

    Args:
        email (str): Email de l'utilisateur
        password (str): Mot de passe en clair
        db (Session): Session de base de données

    Returns:
        User: Objet utilisateur si authentification réussie
        False: Si authentification échouée

    Note:
        Vérifie le hash du mot de passe avec bcrypt
    """
    try:
        statement = select(User).where(User.email == email)
        user = db.exec(statement).first()
        
        if not user:
            return None
            
        if not bcrypt_context.verify(password, user.hashed_password):
            return None
            
        return user
    except Exception as e:
        # Log l'erreur
        print(f"Erreur authentification: {e}")
        return None
    

async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)], db: db_dependency) -> User:
    """
    Récupère l'utilisateur actuel à partir du token JWT.

    Args:
        token (str): Token JWT à décoder
        db (Session): Session de base de données

    Returns:
        User: Utilisateur authentifié

    Raises:
        HTTPException:
            - 401: Si le token est invalide
            - 401: Si l'utilisateur n'existe pas

    Note:
        - Vérifie la validité du token
        - Vérifie l'existence de l'utilisateur
        - Utilisé comme dépendance pour les routes protégées
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None :
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
        statement = select(User).where(User.email == username)
        user = db.exec(statement).first()
        user = db.query(User).filter(User.email == username).first()
        if user is None : 
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nom d'utilisateur invalide")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")

