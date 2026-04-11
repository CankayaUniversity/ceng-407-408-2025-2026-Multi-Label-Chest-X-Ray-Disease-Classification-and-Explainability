from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from database import analyses_collection, patients_collection # patients_collection'ı ekledik
import shutil
import os
from datetime import datetime

router = APIRouter(prefix="/analyze", tags=["Analysis"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_xray(
    doctor_email: str = Form(...),
    protocol_id: str = Form(...),   # patient_no yerine bunu kullanıyoruz
    file: UploadFile = File(...)
):
    # 1. GÜVENLİK KONTROLÜ: Bu ID'ye sahip bir hasta sistemde gerçekten var mı?
    if not patients_collection.find_one({"protocol_id": protocol_id}):
        raise HTTPException(status_code=404, detail="Bu protokol numarasına sahip bir hasta bulunamadı. Lütfen önce hastayı kaydedin.")

    # 2. Sadece resim dosyalarına izin ver
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası yükleyin.")

    # 3. Dosyayı sunucuya kaydet
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. MongoDB'ye kaydedilecek analiz verisi
    analysis_data = {
        "doctor_email": doctor_email,
        "protocol_id": protocol_id,  # Veritabanında da bu isimle tutuyoruz
        "image_path": file_path,
        "status": "pending",
        "ai_result": None,
        "upload_date": datetime.now().strftime("%d %B %Y - %H:%M")
    }

    analyses_collection.insert_one(analysis_data)

    return {
        "message": "Fotoğraf başarıyla yüklendi, analiz bekleniyor.",
        "file_path": file_path
    }