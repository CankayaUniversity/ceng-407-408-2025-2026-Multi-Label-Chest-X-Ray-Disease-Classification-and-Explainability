
from pydantic import BaseModel, EmailStr
from typing import Optional

class RegisterRequest(BaseModel):
    full_name: str    
    email: EmailStr      
    password: str        
    hospital: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PatientCreate(BaseModel):
    full_name: str
    age: int
    gender: str
    hospital: str
    doctor_email: str
    tc_no: Optional[str] = None
    is_foreign: bool = False