from fastapi import APIRouter, HTTPException
import bcrypt
from models import RegisterRequest, LoginRequest
from database import users_collection, db
from bson import ObjectId

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
        "hospital": user.hospital,
        "role": "doctor",
        "is_verified": False,
        "status": "pending"
    }

    users_collection.insert_one(new_user)
    return {"message": "Kayıt talebi alındı. Hastane onayı bekleniyor."}

@router.post("/login")
async def login_user(user: LoginRequest):
    db_user = users_collection.find_one({"email": user.email})
    
    if not db_user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")

    if not bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"]):
        raise HTTPException(status_code=401, detail="Hatalı şifre")

    if not db_user.get("is_verified", False):
        raise HTTPException(
            status_code=403, 
            detail="Hesabınız hastane tarafından henüz onaylanmamıştır."
        )


    return {
        "message": "Giriş başarılı",
        "full_name": db_user["full_name"],
        "role": db_user["role"],
        "hospital": db_user.get("hospital", "")
    }



@router.post("/hospital-login")
@router.post("/hospital-login")
@router.post("/hospital-login")
async def hospital_login(data: dict):

    hosp_name = str(data.get("hospital_name", "")).strip()
    input_password = str(data.get("password", "")).strip()
    

    hosp = db["hospitals"].find_one({"name": hosp_name})
    
    if not hosp:
        raise HTTPException(status_code=404, detail=f"Hastane bulunamadı: {hosp_name}")
    

    db_password = str(hosp.get("password", "")).strip()
    

    print(f"--- LOGIN DENEMESİ ---")
    print(f"Hastane: '{hosp_name}'")
    print(f"Girilen Şifre: '{input_password}'")
    print(f"DB'deki Şifre: '{db_password}'")
    print(f"----------------------")

    if db_password != input_password:
        raise HTTPException(status_code=401, detail="Hatalı hastane şifresi")
        
    return {
        "message": "Hastane girişi başarılı",
        "hospital_name": hosp["name"],
        "role": "hospital_admin"
    }

@router.get("/pending-doctors/{hospital_name}")
async def get_pending_doctors(hospital_name: str):
    docs = list(users_collection.find({"hospital": hospital_name, "is_verified": False}))
    for d in docs:
        d["_id"] = str(d["_id"])
        if "password" in d: del d["password"]
    return docs

@router.get("/approved-doctors/{hospital_name}")
async def get_approved_doctors(hospital_name: str):
    docs = list(users_collection.find({"hospital": hospital_name, "is_verified": True}))
    for d in docs:
        d["_id"] = str(d["_id"])
        if "password" in d: del d["password"]
    return docs

@router.put("/manage-doctor")
async def manage_doctor(data: dict):
    email = data.get("email")
    action = data.get("action")

    if not email or not action:
        raise HTTPException(status_code=400, detail="Email ve action bilgisi gerekli")

    if action == "approve":
        users_collection.update_one(
            {"email": email}, 
            {"$set": {"is_verified": True, "status": "approved"}}
        )
    elif action in ["reject", "delete"]:
        users_collection.delete_one({"email": email})
    
    return {"message": f"İşlem ({action}) başarıyla tamamlandı."}



@router.get("/hospitals")
async def get_hospitals():
    hospitals_data = []
    for hosp in db["hospitals"].find():
        hosp["_id"] = str(hosp["_id"])
        hospitals_data.append(hosp)
    return hospitals_data

@router.get("/init-hospitals")
async def init_hospitals():
    """Hastaneleri veritabanına temizleyip yeniden yükler"""

    db["hospitals"].delete_many({})
    

    hastane_listesi = [
        {"name": "Ankara Bilkent Şehir Hastanesi", "city": "Ankara", "password": "123"},
        {"name": "Hacettepe Üniversitesi Tıp Fakültesi Hastanesi", "city": "Ankara", "password": "123"},
        {"name": "Gazi Üniversitesi Tıp Fakültesi Hastanesi", "city": "Ankara", "password": "123"},
        {"name": "İstanbul Başakşehir Çam ve Sakura Şehir Hastanesi", "city": "İstanbul", "password": "123"},
        {"name": "Ankara Şehir Hastanesi", "city": "Ankara", "password": "123456"}
    ]
    

    db["hospitals"].insert_many(hastane_listesi)
    

    print("DEBUG: Hastaneler ve şifreler DB'ye yüklendi.")
    
    return {"message": "Hastaneler başarıyla yüklendi. Artık giriş yapabilirsiniz."}