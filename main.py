from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = "Agritech")

# Allows app to talk to backend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "Welcome to the Agritech API!"}

@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...),
                   language : str = Form("en")):

    # Here you would add your logic to process the uploaded file and return a diagnosis
    # For now its fake responces, add AI logic later

    responses = {
    "en": {
        "disease": "Early Blight",
        "confidence": 0.78,
        "risk_level": "medium",
        "advice": "Avoid overwatering and remove infected leaves."
    },
    "hi": {
        "disease": "अर्ली ब्लाइट",
        "confidence": 0.78,
        "risk_level": "medium",
        "advice": "ज्यादा पानी न दें और संक्रमित पत्तियां हटा दें।"
    },
    "es": {
        "disease": "Tizón temprano",
        "confidence": 0.78,
        "risk_level": "medium",
        "advice": "Evita el exceso de riego y retira las hojas infectadas."
    }
}
    return responses.get(language, responses["en"])