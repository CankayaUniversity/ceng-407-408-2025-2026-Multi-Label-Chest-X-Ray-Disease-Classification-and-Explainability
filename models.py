# Pydantic class'ların (RegisterRequest vs.) durur

from pydantic import BaseModel, EmailStr
from typing import Optional

class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "doctor"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PatientCreate(BaseModel):
    full_name: str
    age: int
    gender: str
    is_foreign: bool = False  
    tc_no: Optional[str] = None