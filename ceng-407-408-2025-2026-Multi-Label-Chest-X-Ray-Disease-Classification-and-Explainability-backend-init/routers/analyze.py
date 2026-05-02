from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from database import analyses_collection, patients_collection
from bson import ObjectId
import io
import base64
from datetime import datetime
import torch
from PIL import Image
from torchvision import transforms
from ai_models.full_model import DenseNetCBAM
from ai_models.gradCAM import GradCAMPlusPlus
from skimage import measure
from skimage.transform import resize
import os

router = APIRouter(prefix="/analyze", tags=["Analysis"])
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "dense_cbam_weights.pth")

device = "cuda" if torch.torch.cuda.is_available() else "cpu"

model = DenseNetCBAM(num_classes=14)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
target_layer = model.backbone.features[-2]
gradcam = GradCAMPlusPlus(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", 
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", 
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

DISEASE_THRESHOLDS = {
    "Atelectasis": 0.50, "Cardiomegaly": 0.45, "Consolidation": 0.50,
    "Edema": 0.55, "Effusion": 0.55, "Emphysema": 0.65,
    "Fibrosis": 0.50, "Hernia": 0.75, "Infiltration": 0.50,
    "Mass": 0.65, "Nodule": 0.55, "Pleural_Thickening": 0.55,
    "Pneumonia": 0.35, "Pneumothorax": 0.55
}

@router.post("/upload")
async def upload_and_analyze(
    doctor_email: str = Form(...),
    protocol_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not patients_collection.find_one({"protocol_id": protocol_id}):
        raise HTTPException(status_code=404, detail="Bu protokol numarasına sahip bir hasta bulunamadı.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir resim dosyası yükleyin.")
    
    contents = await file.read()
    base64_encoded = base64.b64encode(contents).decode('utf-8')
    base64_image_data = f"data:{file.content_type};base64,{base64_encoded}"
    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    H_orig, W_orig = image.size[1], image.size[0]
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()

    analysis_results = []
    has_any_disease = False

    for idx, name in enumerate(CLASS_NAMES):
        prob = float(probs[idx])
        threshold = DISEASE_THRESHOLDS.get(name, 0.50) 
        
        if prob >= threshold:
            has_any_disease = True
            cam = gradcam(input_tensor, idx, resize_to=(H_orig, W_orig))
            mask = cam >= 0.5 
            mask_resized = resize(mask.astype(float), (1024, 1024))
            cnts = measure.find_contours(mask_resized, 0.5)
            contours = [c.tolist() for c in cnts]
        else:
            contours = []

        analysis_results.append({
            "name": name,
            "prob": prob,
            "contours": contours,
            "threshold_used": threshold
        })

    analysis_data = {
        "doctor_email": doctor_email,
        "protocol_id": protocol_id,
        "original_image": base64_image_data,
        "status": "completed",
        "has_disease": has_any_disease,
        "ai_result": analysis_results,
        "upload_date": datetime.now().strftime("%d %B %Y - %H:%M"),
        "created_at": datetime.now()
    }

    inserted = analyses_collection.insert_one(analysis_data)

    return {
        "id": str(inserted.inserted_id),
        "message": "Analiz başarıyla tamamlandı ve kaydedildi.",
        "has_disease": has_any_disease
    }

@router.get("/result/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı.")
    analysis["_id"] = str(analysis["_id"])
    return analysis

@router.get("/history/{protocol_id}")
async def get_patient_history(protocol_id: str):
    history = list(analyses_collection.find(
        {"protocol_id": protocol_id},
        {"original_image": 0, "ai_result.contours": 0}
    ).sort("created_at", -1))
    
    for item in history:
        item["_id"] = str(item["_id"])
        
    return history

@router.get("/recent/doctor")
async def get_recent_analyses_by_doctor(email: str):
    history = list(analyses_collection.find(
        {"doctor_email": email},
        {"original_image": 0, "ai_result.contours": 0}
    ).sort("created_at", -1).limit(5))
    
    for item in history:
        item["_id"] = str(item["_id"])
        
    return history

@router.post("/finalize-report")
async def finalize_report(
    analysis_id: str = Form(...),
    doctor_comment: str = Form(...),
    doctor_name: str = Form(...),
    hospital_name: str = Form(...)
):
    analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz kaydı bulunamadı.")

    report_data = {
        "report_id": str(ObjectId()),
        "doctor_name": doctor_name,
        "hospital_name": hospital_name,
        "doctor_comment": doctor_comment,
        "finalized_at": datetime.now().strftime("%d %B %Y - %H:%M"),
        "is_official_report": True
    }

    analyses_collection.update_one(
        {"_id": ObjectId(analysis_id)},
        {"$set": {"report": report_data, "status": "finalized"}}
    )

    return {"message": "Rapor başarıyla oluşturuldu ve arşive eklendi.", "report": report_data}