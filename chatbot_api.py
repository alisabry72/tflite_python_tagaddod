from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
import json
import re
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arabic Intent Classification API",
    description="API for classifying Egyptian Arabic intents related to oil collection",
    version="1.0.0"
)

# Load model components
try:
    logger.info("Loading model components...")
    model = tf.keras.models.load_model("advanced_arabic_intent_model.h5")
    with open("advanced_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("advanced_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    with open("training_data.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        intents = data["intents"] if "intents" in data else data
    logger.info("Model components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model components: {str(e)}")
    raise

# Dialogue State Machine
class DialogueStateMachine:
    def __init__(self):
        self.conversations = {}
    
    def get_state(self, user_id):
        return self.conversations.get(user_id, {"state": "STARTED", "context": {}})

    def update_state(self, user_id, new_state, context_update=None):
        current = self.get_state(user_id)
        if context_update:
            current["context"].update(context_update)
        current["state"] = new_state
        self.conversations[user_id] = current
        logger.info(f"User {user_id} state updated: {current}")
        return current

    def guide_back(self, user_id, intent_tag):
        current = self.get_state(user_id)
        state = current["state"]
        context = current["context"]
        if state == "AWAITING_QUANTITY":
            return "تمام يا فندم، ممكن نحدد كمية الزيت عشان نكمل الطلب؟"
        elif state == "AWAITING_ADDRESS":
            return f"حلو، سجلنا الكمية: {context.get('quantity', 'غير محدد')} لتر. ممكن تقول لي عنوانك؟"
        elif state == "AWAITING_GIFT":
            return f"كده عندنا {context.get('quantity', 'غير محدد')} لتر زيت من {context.get('address', 'غير محدد')}. عايز تختار هدية مع الطلب؟ (مثلًا: كوبون خصم)"
        elif state == "AWAITING_CONFIRMATION":
            return f"كده عندنا {context.get('quantity', 'غير محدد')} لتر زيت من {context.get('address', 'غير محدد')} والهدية: {context.get('gift', 'بدون هدية')}. نكمل الطلب؟"
        return "لو عايز تطلب جمع زيت، قول لي كام لتر عندك!"

state_machine = DialogueStateMachine()

class TextInput(BaseModel):
    text: str
    user_id: str

class ArabicJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")

def predict_intent(text: str, user_id: str):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
        prediction = model.predict(padded_sequence, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        intent_tag = encoder.inverse_transform([predicted_class_idx])[0]

        current_state = state_machine.get_state(user_id)
        state = current_state["state"]
        context = current_state["context"]
        response = None

        # Core state machine logic
        if intent_tag == "greeting" and state == "STARTED":
            response = "وعليكم السلام يا فندم! كام لتر زيت عندك عشان نجمعهم؟"
            current_state = state_machine.update_state(user_id, "AWAITING_QUANTITY")

        elif intent_tag == "provide_quantity" and state == "AWAITING_QUANTITY":
            match = re.search(r'\d+', text)
            amount = match.group(0) if match else None
            if amount:
                context["quantity"] = amount
                response = f"تمام، سجلنا الكمية: {amount} لتر. ممكن تقول لي عنوانك الكامل؟"
                current_state = state_machine.update_state(user_id, "AWAITING_ADDRESS", {"quantity": amount})
            else:
                response = "مفهمتش الكمية، ممكن تقول لي كام لتر بالظبط؟"

        elif intent_tag == "provide_address" and state == "AWAITING_ADDRESS":
            context["address"] = text.strip()
            response = f"حلو، سجلنا العنوان: {text}. عايز تختار هدية مع الطلب؟ (مثلًا: كوبون خصم)"
            current_state = state_machine.update_state(user_id, "AWAITING_GIFT", {"address": text})

        elif intent_tag == "choose_gift" and state == "AWAITING_GIFT":
            gift = text.strip()
            context["gift"] = gift
            response = f"تمام، اخترنا الهدية: {gift}. كده عندنا {context.get('quantity', 'غير محدد')} لتر زيت من {context.get('address', 'غير محدد')}. نكمل الطلب؟"
            current_state = state_machine.update_state(user_id, "AWAITING_CONFIRMATION", {"gift": gift})

        elif intent_tag == "submit_order" and state == "AWAITING_CONFIRMATION":
            response = f"تم تسجيل طلبك بنجاح! هنيجي نجمع {context.get('quantity', 'غير محدد')} لتر زيت من {context.get('address', 'غير محدد')} والهدية: {context.get('gift', 'بدون هدية')}. شكرًا يا فندم!"
            current_state = state_machine.update_state(user_id, "STARTED")

        elif intent_tag == "choose_gift" and state in ["STARTED", "AWAITING_QUANTITY"]:
            gift = text.strip()
            response = f"تمام، اخترنا الهدية: {gift}. لو عايز تطلب زيت معاها، قول لي الكمية!"
            current_state = state_machine.update_state(user_id, "AWAITING_QUANTITY", {"gift": gift})

        elif intent_tag == "ask_for_help":
            response = "تحت أمرك! لو محتاج مساعدة في حاجة معينة، قول لي وأنا أساعدك."

        # Guide back to context if no specific response
        if not response:
            response = state_machine.guide_back(user_id, intent_tag)
            current_state = state_machine.get_state(user_id)

        logger.info(f"Returning response for {user_id}: {response}")
        return {
            "intent": intent_tag,
            "confidence": confidence,
            "response": response,
            "state": current_state["state"]
        }
    except Exception as e:
        logger.error(f"Error in predict_intent: {str(e)}")
        raise

@app.post("/predict", response_class=ArabicJSONResponse)
async def predict(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        result = predict_intent(input.text, input.user_id)
        return {
            "status": "success",
            "data": {
                "original_text": input.text,
                "state": result["state"],
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
                "response": result["response"]
            }
        }
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/", response_class=ArabicJSONResponse)
async def root():
    return {"message": "Welcome to Arabic Intent Classification API"}

@app.get("/health", response_class=ArabicJSONResponse)
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)