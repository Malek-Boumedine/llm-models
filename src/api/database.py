from sqlmodel import SQLModel, create_engine, text, Session
from dotenv import load_dotenv
import os
from src.api.models import *


load_dotenv()

API_DATABASE_HOST = os.getenv("API_DATABASE_HOST", None)
API_DATABASE_PORT = os.getenv("API_DATABASE_PORT", None)
API_DATABASE_NAME = os.getenv("API_DATABASE_NAME", None)
API_DATABASE_USERNAME = os.getenv("API_DATABASE_USERNAME", None)
API_DATABASE_ROOT_PASSWORD = os.getenv("API_DATABASE_ROOT_PASSWORD", None)

DATABASE_URL = f"postgresql://{API_DATABASE_USERNAME}:{API_DATABASE_ROOT_PASSWORD}@{API_DATABASE_HOST}:{API_DATABASE_PORT}/{API_DATABASE_NAME}"
engine = create_engine(DATABASE_URL, echo=True)
SQLModel.metadata.create_all(engine)

def db_connection(): 
    session = Session(engine)
    try: 
        yield session
    finally: 
        session.close()
        
