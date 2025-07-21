from sqlmodel import SQLModel, create_engine, text, Session
from dotenv import load_dotenv
import os
from src.chainlit_app.models import *


load_dotenv()

CHAINLIT_DATABASE_HOST = os.getenv("CHAINLIT_DATABASE_HOST", None)
CHAINLIT_DATABASE_PORT = os.getenv("CHAINLIT_DATABASE_PORT", None)
CHAINLIT_DATABASE_NAME = os.getenv("CHAINLIT_DATABASE_NAME", None)
CHAINLIT_DATABASE_USERNAME = os.getenv("CHAINLIT_DATABASE_USERNAME", None)
CHAINLIT_DATABASE_ROOT_PASSWORD = os.getenv("CHAINLIT_DATABASE_ROOT_PASSWORD", None)

DATABASE_URL = f"postgresql://{CHAINLIT_DATABASE_USERNAME}:{CHAINLIT_DATABASE_ROOT_PASSWORD}@{CHAINLIT_DATABASE_HOST}:{CHAINLIT_DATABASE_PORT}/{CHAINLIT_DATABASE_NAME}"
engine = create_engine(DATABASE_URL, echo=True)
SQLModel.metadata.create_all(engine)

def db_connection(): 
    session = Session(engine)
    try: 
        yield session
    finally: 
        session.close()
        
