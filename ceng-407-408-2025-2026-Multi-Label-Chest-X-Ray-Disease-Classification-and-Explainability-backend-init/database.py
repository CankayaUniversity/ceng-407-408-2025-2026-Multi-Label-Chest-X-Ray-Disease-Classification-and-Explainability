
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://salihbarkinakkaya:salih123@ceng408.55cbswg.mongodb.net/?retryWrites=true&w=majority&appName=Ceng408"
client = MongoClient(MONGO_URI)

db = client["ceng408"]
users_collection = db["users"]
analyses_collection = db["analyses"]
patients_collection = db["patients"]