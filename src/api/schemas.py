from pydantic import BaseModel, Field, EmailStr



class AskQuestion(BaseModel): 
    
    question: str = Field(..., description="Énoncé")
    class Config: 
        json_schema_extra = {
            "Example" : {
                "question" : "Quel est le montant du SMIC mensuel selon le code du travail ?"
            }
        }


class CreateAccount(BaseModel): 
    username: str
    email: EmailStr
    password: str = Field(..., min_length=8, description="longueur minimale de 8 caractères")
    password_confirm: str = Field(..., min_length=8, description="longueur minimale de 8 caractères")
    class Config: 
        json_schema_extra = {
            "Example": {
                "username" : "lala.kiki",
                "email" : "lala.kiki@email.com",
                "password" : "azerty12", 
                "password_confirm" : "azerty12"
            }
        }



