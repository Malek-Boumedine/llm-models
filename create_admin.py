from sqlmodel import Session, select
from passlib.context import CryptContext
from src.chainlit_app.database import engine
from src.chainlit_app.models import User
import os
from dotenv import load_dotenv

load_dotenv()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@admin.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Azerty1234")

bcrypt = CryptContext(schemes=["bcrypt"], deprecated="auto")

username = ADMIN_USERNAME

with Session(engine) as db:
    admin = User(
        username=username,
        email=ADMIN_EMAIL, 
        hashed_password=bcrypt.hash(ADMIN_PASSWORD),
        role="admin"
    )
    
    existing_user = db.exec(select(User).where(User.username == username)).first()
    
    if existing_user is None:
        db.add(admin)
        db.commit()
        print(f"✅ {username} créé : admin/admin")
    else:
        print(f"L'utilisateur {username} existe déjà")
