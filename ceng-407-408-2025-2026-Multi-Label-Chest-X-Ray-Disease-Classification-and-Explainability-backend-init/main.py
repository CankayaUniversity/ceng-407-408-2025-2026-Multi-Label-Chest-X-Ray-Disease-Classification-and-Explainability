from fastapi import FastAPI
from routers import auth, analyze, patients
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ChestXplain API")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(auth.router)
app.include_router(analyze.router) 
app.include_router(patients.router)

@app.get("/")
def root():
    return {"status": "Sistem Calisiyor", "project": "ChestXplain Backend"}

