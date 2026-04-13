from fastapi import APIRouter, HTTPException
import bcrypt
from models import RegisterRequest, LoginRequest
from database import users_collection
from database import users_collection, db

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
async def register_user(user: RegisterRequest):
    # 1. E-posta kontrolü
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Bu e-posta zaten kayıtlı")

    # 2. Şifreyi şifreleme
    hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    # 3. Yeni kullanıcı objesini oluştur (Frontend'den gelen verilerle)
    new_user = {
        "full_name": user.name_surname,   
        "email": user.email,
        "password": hashed_pw,
        "hospital": user.hospital,        
        "role": "doctor",                 
        "is_verified": False,
        "status": "pending"
    }

    # 4. Veritabanına kaydet
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
        "role": db_user["role"],
        "hospital": db_user.get("hospital", "") # Giriş yaparken hastane bilgisini de geri dönüyoruz
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

@router.get("/hospitals")
async def get_hospitals():
    hospitals_data = []
    
    # Veritabanındaki tüm hastaneleri çekiyoruz
    for hosp in db["hospitals"].find():
        # En kritik satır: MongoDB'nin kendi oluşturduğu _id'yi 
        # React Native'in okuyabileceği normal bir metne çeviriyoruz!
        hosp["_id"] = str(hosp["_id"]) 
        hospitals_data.append(hosp)
        
    return hospitals_data

@router.get("/add-hospitals")
async def add_hospitals():
    """Kendi veritabanımıza hastane verileri ekler"""
    hastane_listesi = [
        {"name": "Ankara Bilkent Şehir Hastanesi", "city": "Ankara"},
        {"name": "Hacettepe Üniversitesi Tıp Fakültesi Hastanesi", "city": "Ankara"},
        {"name": "Gazi Üniversitesi Tıp Fakültesi Hastanesi", "city": "Ankara"},
        {"name": "İstanbul Başakşehir Çam ve Sakura Şehir Hastanesi", "city": "İstanbul"},
        {"name": "İzmir Şehir Hastanesi", "city": "İzmir"}
    ]
    
    # MongoDB "hospitals" adında bir tabloyu otomatik açar ve bu listeyi içine basar
    db["hospitals"].insert_many(hastane_listesi)

    