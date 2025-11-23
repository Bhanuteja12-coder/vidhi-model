from fastapi import FastAPI
from pydantic import BaseModel
import joblib, re

app = FastAPI()

# Load models
clf_domain = joblib.load("models/domain_model.pkl")
clf_urgency = joblib.load("models/urgency_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
le = joblib.load("models/label_encoder.pkl")

class Query(BaseModel):
    query: str

@app.post("/predict")
def predict(q: Query):
    query_clean = re.sub(r'[^a-zA-Z\s]', '', q.query.lower())
    query_vec = vectorizer.transform([query_clean])
    domain_pred = clf_domain.predict(query_vec)[0]
    domain_label = le.inverse_transform([domain_pred])[0]
    urgency_pred = clf_urgency.predict(query_vec)[0]
    return {"domain": domain_label, "urgency": int(urgency_pred)}
