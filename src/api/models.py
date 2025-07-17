from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, String
from sqlalchemy import DateTime
from sqlmodel import Relationship
from typing import List




class User(SQLModel, table=True): 
    user_id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String(255), unique=True))
    email: str = Field(sa_column=Column(String(255), unique=True))
    hashed_password: str = Field(sa_column=Column(String(255)))
    role: str = Field(default="user", sa_column=Column(String(50)))
    created_at: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime))
    
    # Relations
    questions: List["Question"] = Relationship(back_populates="user")
    answers: List["Answer"] = Relationship(back_populates="user")
    

class Question(SQLModel, table=True):
    question_id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    question_date: datetime = Field(default_factory=datetime.now)
    user_id: int = Field(foreign_key="user.user_id")
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.conversation_id")


class Answer(SQLModel, table=True):
    answer_id: Optional[int] = Field(default=None, primary_key=True)
    answer: str
    answer_date: datetime = Field(default_factory=datetime.now)
    user_id: int = Field(foreign_key="user.user_id")
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.conversation_id")


class Conversation(SQLModel, table=True):
    conversation_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.user_id")  # Ajouter cette ligne
    creation_date: datetime = Field(default_factory=datetime.now)
    
    # Relations
    questions: List["Question"] = Relationship(back_populates="conversation")
    answers: List["Answer"] = Relationship(back_populates="conversation")

