from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils.exercise_model import predict_exercise
from utils.food_model import predict_food
import os
import uuid
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI()

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

class UserInput(BaseModel):
    tinggi: float
    berat: float
    durasi: float
    kalori_terbakar: float
    umur: float
    tingkat_aktivitas: str
    tujuan: str
    kategori: str
    jenis_kelamin: str

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
        'Tinggi (cm)': user_input.tinggi,
        'Berat (kg)': user_input.berat,
        'Durasi (menit)': user_input.durasi,
        'Kalori Terbakar': user_input.kalori_terbakar,
        'Umur': user_input.umur,
        'Tingkat Aktivitas': user_input.tingkat_aktivitas,
        'Tujuan': user_input.tujuan,
        'Kategori': user_input.kategori,
        'Jenis Kelamin': user_input.jenis_kelamin
    }])

    recommended_exercise = predict_exercise(input_data)
    return {"recommended_exercise": recommended_exercise}
