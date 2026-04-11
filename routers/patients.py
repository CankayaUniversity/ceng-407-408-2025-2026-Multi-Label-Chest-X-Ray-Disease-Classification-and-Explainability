from fastapi import APIRouter, HTTPException
from database import patients_collection
from models import PatientCreate
import random

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.post("/add")
async def add_patient(patient: PatientCreate):
    # 1. T.C. No Kontrolü (Eğer girilmişse)
    if patient.tc_no:
        existing_patient = patients_collection.find_one({"tc_no": patient.tc_no})
        if existing_patient:
            raise HTTPException(status_code=400, detail="Bu T.C. numarası ile kayıtlı bir hasta zaten var.")

    # 2. Sistemin Atayacağı Benzersiz Protokol ID Üretimi
    while True:
        # Örn: P-102534 formatında bir ID
        new_protocol_id = f"P-{random.randint(100000, 999999)}"
        if not patients_collection.find_one({"protocol_id": new_protocol_id}):
            break

    # 3. Veriyi Hazırla ve Kaydet
    patient_data = patient.dict()
    patient_data["protocol_id"] = new_protocol_id # Sistemin atadığı ID
    
    patients_collection.insert_one(patient_data)
    
    return {
        "message": "Hasta başarıyla sisteme kaydedildi.",
        "protocol_id": new_protocol_id,
        "full_name": patient.full_name
    }