# Arabic Intent Classification API - Technical Architecture

## System Architecture

The backend is built using FastAPI and implements a conversational state machine to manage multi-turn dialogues in Egyptian Arabic. This document outlines the technical details of the implementation.

## Core Components

### 1. DialogueStateMachine

The `DialogueStateMachine` class is the central component that manages conversation state:

```python
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
        return current
    
    def clear_session(self, user_id):
        if user_id in self.conversations:
            del self.conversations[user_id]
            return True
        return False

    def guide_back(self, user_id, intent_tag):
        # Logic to guide users back to their current conversation step
        # ...
```

#### Key Features:

- **In-memory Session Storage**: Uses a dictionary (`self.conversations`) to store user sessions
- **State Management**: Maintains the current state of each conversation
- **Context Storage**: Stores contextual information (quantity, address, etc.)
- **Session Clearing**: Allows removing a user's session when conversation ends

### 2. Intent Classification

The system uses a TensorFlow model to classify user intents:

```python
def predict_intent(text: str, user_id: str):
    # Preprocess text for model
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    
    # Get model prediction
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class_idx])
    intent_tag = encoder.inverse_transform([predicted_class_idx])[0]
    
    # Get current state and process based on intent
    # ...
```

#### Key Features:

- **Text Preprocessing**: Tokenization and padding
- **Intent Classification**: TensorFlow model prediction
- **Confidence Scoring**: Probability score for predicted intent

### 3. State Machine Logic

The conversation follows a strict flow controlled by states:

```
STARTED → AWAITING_QUANTITY → AWAITING_ADDRESS → AWAITING_GIFT → AWAITING_CONFIRMATION → (Order Complete) → STARTED
```

The logic enforces this flow in the `predict_intent` function:

```python
# Core state machine logic
if intent_tag == "greeting" and state == "STARTED":
    response = "وعليكم السلام يا فندم! كام لتر زيت عندك عشان نجمعهم؟"
    current_state = state_machine.update_state(user_id, "AWAITING_QUANTITY")

# Enforce strict flow - user must provide quantity first
elif state == "STARTED" or state == "AWAITING_QUANTITY":
    # If the user sent a provide_quantity intent, process it
    if intent_tag == "provide_quantity":
        match = re.search(r'\d+', text)
        amount = match.group(0) if match else None
        if amount:
            context["quantity"] = amount
            response = f"تمام، سجلنا الكمية: {amount} لتر. ممكن تقول لي عنوانك الكامل؟"
            current_state = state_machine.update_state(user_id, "AWAITING_ADDRESS", {"quantity": amount})
        else:
            response = "مفهمتش الكمية، ممكن تقول لي كام لتر بالظبط؟"
            current_state = state_machine.update_state(user_id, "AWAITING_QUANTITY")
    # If it's any other intent, keep asking for quantity
    else:
        response = "نحتاج نعرف كمية الزيت الأول. ممكن تقول لي كام لتر عندك؟"
        current_state = state_machine.update_state(user_id, "AWAITING_QUANTITY")
```

## Conversation Flow Details

### State: STARTED
- Initial state for new users or after completing an order
- Valid transitions:
  - To AWAITING_QUANTITY: After greeting or any message

### State: AWAITING_QUANTITY
- System is waiting for the user to provide oil quantity
- Key features:
  - Enforces providing quantity before proceeding
  - Uses regex pattern `r'\d+'` to extract numeric values
  - Stores quantity in context when detected
- Valid transitions:
  - To AWAITING_ADDRESS: After successfully extracting quantity
  - Stays in AWAITING_QUANTITY: If quantity not detected or other intents recognized

### State: AWAITING_ADDRESS
- System is waiting for address information
- Key features:
  - Stores entire text as address
- Valid transitions:
  - To AWAITING_GIFT: After receiving any address input

### State: AWAITING_GIFT
- System is waiting for optional gift selection
- Key features:
  - Stores gift selection in context
- Valid transitions:
  - To AWAITING_CONFIRMATION: After receiving gift choice

### State: AWAITING_CONFIRMATION
- System is waiting for order confirmation
- Valid transitions:
  - To STARTED: After receiving confirmation (order complete)

## Technical Decisions and Reasoning

### In-memory Session Storage
- **Pros**: Simple implementation, fast access
- **Cons**: Not persistent across server restarts
- **Reasoning**: Suitable for prototyping and testing; for production, consider a persistent store like Redis

### Regex for Quantity Extraction
- Uses regular expression `r'\d+'` to find numeric values in text
- Simple but effective for basic number extraction
- For more complex parsing, consider a specialized NLP entity extraction approach

### Strict Conversation Flow
- Forces users through a specific sequence
- Prevents skipping steps in the ordering process
- Guarantees all required information is collected

## API Endpoint Implementations

### Predict Endpoint
```python
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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
```

### End Session Endpoint
```python
@app.post("/end_session", response_class=ArabicJSONResponse)
async def end_session(input: SessionInput):
    """
    Endpoint to clear a user's session when they leave the chat screen
    """
    success = state_machine.clear_session(input.user_id)
    if success:
        return {"status": "success", "message": "User session cleared successfully"}
    else:
        return {"status": "success", "message": "No active session found for this user"}
```

## Performance Considerations

1. **Memory Usage**: As more users interact with the system, memory usage will increase. Consider implementing session expiration.

2. **Model Inference**: Prediction could become a bottleneck with many concurrent users. Consider:
   - Model optimization/quantization
   - Caching common responses
   - Horizontal scaling

3. **Session Management**: For production, implement:
   - Session timeout mechanism
   - Persistent storage (Redis, database)

## Security Considerations

1. **User Authentication**: Currently no authentication is implemented
   - Consider adding JWT token authentication for production

2. **Input Validation**: Input is minimally validated
   - Add more robust input validation and sanitization

3. **Rate Limiting**: No rate limiting is implemented
   - Add rate limiting to prevent abuse

## Future Architecture Improvements

1. **Persistent Storage**: Move from in-memory storage to a database like MongoDB or Redis
   - Allows scaling horizontally
   - Maintains state across server restarts

2. **Enhanced NLP**: Improve entity extraction
   - Use named entity recognition for addresses
   - Implement better quantity extraction

3. **Containerization**: Package the application as a Docker container
   - Simplifies deployment and scaling
   - Ensures consistent environment

4. **Monitoring**: Add comprehensive logging and monitoring
   - Track conversation success rates
   - Monitor model performance

5. **A/B Testing**: Implement framework for testing different response patterns
   - Improve conversation success rate
   - Optimize user experience 