# **PolaFit - A Digital Companion for a Healthy Lifestyle**

## **Description**  
PolaFit is an app designed to address the challenges of achieving a sustainable healthy lifestyle, from challenges such as lack of advice personalized to the user, time constraints, and difficulty maintaining motivation. Many people find it difficult to balance their health goals of losing weight, keeping fit, or gaining muscle mass because the advice provided is often too general, progress tracking is unclear, and programs are too rigid to meet their specific needs. PolaFit solves this problem by providing nutrition and exercise recommendations tailored to the user's health profile, preferences, and goals. It uses data-driven algorithms, including Convolutional Neural Networks (CNN) approach to analyze food images that support a variety of visual analysis-based features such as auto-detection, accurate nutrition tracking, and real-time exercise feedback that enhance user experience through accuracy and personalization of services.


## **Installation**

### Prerequisites
Make sure you have **Python 3.x** installed. You also need to install the dependencies listed in the `requirements.txt` file.

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/polafit.git
   cd polafit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Dataset**  
The app uses datasets such as **Food-101**  for food image classification. The dataset can be downloaded from their respective sources:

- [Food-101 Dataset](https://www.kaggle.com/datasets/dansbecker/food-101) 


## **Model Architecture**  
PolaFit uses DenseNet121 for food image analysis and personalized recommendation algorithms for exercise and nutrition plans. The model is fine-tuned using a two-stage tuning technique:

-In the first stage, the Dense layers are frozen (set to False), meaning they are not trained.
-In the second stage, the Dense layers are unfrozen (set to True), allowing them to be trained for better accuracy.

This approach helps improve the model's performance by first leveraging pre-trained weights and then fine-tuning it for specific tasks related to food classification.


Here’s the updated section with the change you requested:

## **Training the Model**  
To train the model, simply open the Jupyter notebook and run all the cells. The notebook will handle the training process, including setting parameters like the number of epochs, batch size, and learning rate.

Make sure to adjust the training parameters directly in the notebook if needed.

## **Evaluation**  
The model is evaluated using **accuracy**, **precision**, and **recall**. You can evaluate the model's performance with the following command:

```bash
python evaluate.py --model food_model_20.h5 
```
#change to recomendation_sportTf_model.h5 for excersie recomendation models

This will display metrics such as accuracy and loss, helping to assess the performance of the model.

Here's the updated **Usage** section with the provided FastAPI code:

---

## **Usage**  
PolaFit's food image analysis and exercise recommendation system can be accessed through the following endpoints:

### **1. Predict Food from Image**  
Use the `/predict_food` endpoint to classify food from an image. Send an image file as input.

```bash
POST /predict_food
```

**Request**  
- Upload an image file using the `file` parameter.

**Response**  
- Returns nutrition information related to the food in the image.

```python
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
```

### **2. Predict Exercise Recommendation**  
Use the `/predict_exercise` endpoint to get personalized exercise recommendations based on user input.

```bash
POST /predict_exercise
```

**Request**  
- Send a JSON body with the following parameters:
  - `tinggi` (Height in cm)
  - `berat` (Weight in kg)
  - `durasi` (Exercise duration in minutes)
  - `kalori_terbakar` (Calories burned)
  - `umur` (Age)
  - `tingkat_aktivitas` (Activity level)
  - `tujuan` (Goal)
  - `kategori` (Category)
  - `jenis_kelamin` (Gender)

Example request body:
```json
{
  "tinggi": 175,
  "berat": 70,
  "durasi": 45,
  "kalori_terbakar": 500,
  "umur": 25,
  "tingkat_aktivitas": "High",
  "tujuan": "Weight Loss",
  "kategori": "Beginner",
  "jenis_kelamin": "Male"
}
```

**Response**  
- Returns recommended exercises based on the input data.

```python
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
```

---

## **Results**  
Sample results after training:

- **Food Classification:**
```json
  {
    "ID": 8,
    "Makanan": "Hamburger",
    "Berat per Serving (g)": "200",
    "Kalori (kcal)": "300",
    "Protein (g)": "20",
    "Lemak (g)": "18",
    "Karbohidrat (g)": "25",
    "Serat (g)": "3",
    "Gula (g)": "5"
}
```

- **Exercise Recommendations:**  
  - Suggested exercises: Bersepeda, Angkat Beban, Senam

