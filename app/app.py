from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.exercise_model import predict_exercise
from utils.food_model import predict_food
import os
import uuid
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UserInput(BaseModel):
    height: float
    weight: float
    duration: float
    calori: float
    age: float
    activity: str
    goal: str
    category: str
    gender: str

@app.post("/predict_food")
async def predict(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        nutrition_info = predict_food(file_location)
        os.remove(file_location)
        
        return JSONResponse(content=nutrition_info)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_exercise")
def recommend_exercise_endpoint(user_input: UserInput):
    input_data = pd.DataFrame([{
        'Tinggi (cm)': user_input.height,
        'Berat (kg)': user_input.weight,
        'Durasi (menit)': user_input.duration,
        'Kalori Terbakar': user_input.calori,
        'Umur': user_input.age,
        'Tingkat Aktivitas': user_input.activity,
        'Tujuan': user_input.goal,
        'Kategori': user_input.category,
        'Jenis Kelamin': user_input.gender
    }])

    recommended_exercise = predict_exercise(input_data)
    return {"recommended_exercise": recommended_exercise}
