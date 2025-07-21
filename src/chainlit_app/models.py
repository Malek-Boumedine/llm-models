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
    
    # relations
    messages: List["Message"] = Relationship(back_populates="user")
    conversations: List["Conversation"] = Relationship(back_populates="user") 
    

class Message(SQLModel, table=True):
    message_id: Optional[int] = Field(default=None, primary_key=True)
    message: str
    message_date: datetime = Field(default_factory=datetime.now)
    message_type: str = Field(default="user")  # "user" ou "assistant"
    user_id: int = Field(foreign_key="user.user_id")
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.conversation_id")

    # relarions
    user: Optional["User"] = Relationship(back_populates="messages")
    conversation: Optional["Conversation"] = Relationship(back_populates="messages")
    

class Conversation(SQLModel, table=True):
    conversation_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.user_id")  # Ajouter cette ligne
    creation_date: datetime = Field(default_factory=datetime.now)
    
    # Relations
    user: Optional["User"] = Relationship(back_populates="conversations")
    messages: List["Message"] = Relationship(back_populates="conversation")
