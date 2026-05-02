from fastapi import APIRouter, HTTPException, Query
from database import patients_collection
from models import PatientCreate
import random
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/patients", tags=["Patients"])

@router.get("")
async def get_all_patients(email: Optional[str] = Query(None)):
    patients_data = []
    

    query = {}
    if email:
        query = {"doctor_email": email}
        

    for patient in patients_collection.find(query):

        patient["_id"] = str(patient["_id"]) 
        patients_data.append(patient)
        
    return patients_data


@router.post("/add")
async def add_patient(patient: PatientCreate):

    if patient.tc_no:
        existing_patient = patients_collection.find_one({
            "tc_no": patient.tc_no, 
            "doctor_email": patient.doctor_email
        })
        if existing_patient:
            raise HTTPException(
                status_code=400, 
                detail="Bu T.C. numarası ile sizin listenizde kayıtlı bir hasta zaten var."
            )


    while True:
        new_protocol_id = f"P-{random.randint(100000, 999999)}"

        if not patients_collection.find_one({"protocol_id": new_protocol_id}):
            break


    patient_data = patient.dict()
    patient_data["protocol_id"] = new_protocol_id
    patient_data["created_at"] = datetime.now()
    

    result = patients_collection.insert_one(patient_data)

    return {
        "message": "Hasta başarıyla sisteme kaydedildi.",
        "protocol_id": new_protocol_id,
        "full_name": patient.full_name,
        "id": str(result.inserted_id)
    }


@router.get("/{protocol_id}/history")
async def get_patient_history(protocol_id: str):

    history = list(db["analyses"].find(
        {"protocol_id": protocol_id}, 
        {"_id": 1, "created_at": 1, "results": 1}
    ).sort("created_at", -1))

    for record in history:
        record["_id"] = str(record["_id"])
        
    return history