from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, String
from sqlalchemy import DateTime




class Users(SQLModel, table=True): 
    user_id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String(255), unique=True))
    email: str = Field(sa_column=Column(String(255), unique=True))
    hashed_password: str = Field(sa_column=Column(String(255)))
    role: str = Field(default="user", sa_column=Column(String(50)))
    created_at: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime))


class Question(SQLModel, table=True):
    question_id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    question_date: datetime = Field(default_factory=datetime.now)
    user_id: int = Field(foreign_key="users.user_id")
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.conversation_id")


class Reponse(SQLModel, table=True):
    response_id: Optional[int] = Field(default=None, primary_key=True)
    reponse: str
    response_date: datetime = Field(default_factory=datetime.now)
    user_id: int = Field(foreign_key="users.user_id")
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.conversation_id")


class Conversation(SQLModel, table=True):
    conversation_id: Optional[int] = Field(default=None, primary_key=True)
    creation_date: datetime = Field(default_factory=datetime.now)


