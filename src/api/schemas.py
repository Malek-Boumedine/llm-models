from pydantic import BaseModel, Field, EmailStr, ConfigDict


class AskQuestion(BaseModel): 
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Quel est le montant du SMIC mensuel selon le code du travail ?"
            }
        }
    )
    
    question: str = Field(..., description="Énoncé")


class CreateAccount(BaseModel): 
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "lala.kiki",
                "email": "lala.kiki@email.com",
                "password": "azerty12", 
                "password_confirm": "azerty12"
            }
        }
    )
    
    username: str
    email: EmailStr
    password: str = Field(..., min_length=8, description="longueur minimale de 8 caractères")
    password_confirm: str = Field(..., min_length=8, description="longueur minimale de 8 caractères")
    role: str = Field(default="user", pattern="^(user|admin)$")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QuestionResponse(BaseModel):
    answer: str
    conversation_id: int
