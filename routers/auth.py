# Sadece Login, Register, Approve kodları

from fastapi import APIRouter, HTTPException
import bcrypt
from models import RegisterRequest, LoginRequest
from database import users_collection

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
async def register_user(user: RegisterRequest):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Bu e-posta zaten kayıtlı")

    hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    new_user = {
        "full_name": user.full_name,
        "email": user.email,
        "password": hashed_pw,
        "role": user.role,
        "is_verified": False,
        "status": "pending"
    }

    users_collection.insert_one(new_user)
    return {"message": "Kayıt talebiniz alındı. Yönetici onayı bekleniyor."}

@router.post("/login")
async def login_user(user: LoginRequest):
    # 1. Veritabanında bu e-postaya sahip kullanıcıyı bul
    db_user = users_collection.find_one({"email": user.email})
    
    if not db_user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")

    # 2. Şifrenin doğruluğunu kontrol et (Bcrypt ile)
    if not bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"]):
        raise HTTPException(status_code=401, detail="Hatalı şifre")

    # 3. Yönetici onayı (is_verified) kontrolü
    if not db_user.get("is_verified", False):
        raise HTTPException(
            status_code=403, 
            detail="Hesabınız henüz yönetici tarafından onaylanmamıştır."
        )

    # 4. Her şey tamamsa giriş başarılı
    return {
        "message": "Giriş başarılı",
        "full_name": db_user["full_name"],
        "role": db_user["role"]
    }

@router.put("/approve/{email}")
async def approve_doctor(email: str):
    # Kullanıcının is_verified ve status alanlarını güncelle
    result = users_collection.update_one(
        {"email": email},
        {"$set": {"is_verified": True, "status": "approved"}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
        
    return {"message": f"{email} hesabı başarıyla onaylandı."}