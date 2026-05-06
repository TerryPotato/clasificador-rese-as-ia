import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
print("⏳ Loading model...")
MODEL_PATH = "./flan-t5-small-sentiment-model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model = model.to("cuda")
model.eval()
print("✅ Model ready!")

class ReviewRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(req: ReviewRequest):
    prompt = f"Classify the sentiment of this movie review as 'positive' or 'negative':\n\n{req.text}"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    prediction = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
    
    scores = torch.softmax(outputs.scores[0][0], dim=-1)
    positive_id = tokenizer.encode("positive")[0]
    negative_id = tokenizer.encode("negative")[0]
    pos_score = scores[positive_id].item()
    neg_score = scores[negative_id].item()
    total = pos_score + neg_score
    
    if "positive" in prediction:
        sentiment = "POSITIVE"
        confidence = (pos_score / total) * 100
    elif "negative" in prediction:
        sentiment = "NEGATIVE"
        confidence = (neg_score / total) * 100
    else:
        sentiment = "UNDEFINED"
        confidence = 0
    
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 1),
        "pos_score": round((pos_score / total) * 100, 1),
        "neg_score": round((neg_score / total) * 100, 1)
    }

app.mount("/", StaticFiles(directory=".", html=True), name="static")