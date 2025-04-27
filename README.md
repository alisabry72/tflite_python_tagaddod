# Arabic Intent Classification API

## Overview
This project implements a conversational API for Egyptian Arabic intent classification, specifically designed for an oil collection service. The API processes natural language input, classifies intents, and maintains conversation context through a state machine.

## Features
- Arabic natural language understanding
- Contextual conversation handling via state machine
- Intent classification with confidence scores
- Support for multi-step conversations
- Session management for user interactions

## Technical Stack
- FastAPI for the REST API framework
- TensorFlow for the NLP model
- Pydantic for data validation
- scikit-learn for preprocessing

## Setup and Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- FastAPI

### Installation
1. Clone the repository
```bash
git clone [repository-url]
cd [repository-dir]
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Ensure you have the following files in your directory:
   - `advanced_arabic_intent_model.h5`
   - `advanced_tokenizer.pkl`
   - `advanced_encoder.pkl`
   - `max_len.pkl`
   - `training_data.json`

### Running the Application
```bash
python chatbot_api.py
```
Or using uvicorn directly:
```bash
uvicorn chatbot_api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Predict Intent
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "سلام عليكم",
    "user_id": "unique_user_identifier"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "original_text": "سلام عليكم",
      "state": "AWAITING_QUANTITY", 
      "predicted_intent": "greeting",
      "confidence": 0.9996733665466309,
      "response": "وعليكم السلام يا فندم! كام لتر زيت عندك عشان نجمعهم؟"
    }
  }
  ```

### 2. End Session
- **URL**: `/end_session`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "user_id": "unique_user_identifier"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success", 
    "message": "User session cleared successfully"
  }
  ```

### 3. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy", 
    "message": "API is running"
  }
  ```

## Architecture

### Dialogue State Machine
The core of the system is a state machine that tracks conversation context:

- **STARTED**: Initial state
- **AWAITING_QUANTITY**: Waiting for the user to provide oil quantity
- **AWAITING_ADDRESS**: Waiting for the user to provide their address
- **AWAITING_GIFT**: Waiting for the user to select an optional gift
- **AWAITING_CONFIRMATION**: Waiting for the user to confirm the order

The system enforces a strict flow where users must provide a quantity before proceeding to other steps.

### Intent Classification
The system uses a TensorFlow model to classify user intents:

- **greeting**: Greetings and salutations
- **provide_quantity**: Information about oil quantity
- **provide_address**: Location information
- **choose_gift**: Gift selection
- **submit_order**: Order confirmation
- **ask_for_help**: Help requests

### Session Management
- Sessions are identified by a unique `user_id`
- Session context is maintained in memory
- The `/end_session` endpoint allows clearing a session (e.g., when a user exits a chat interface)

## Conversation Flow Example

1. User: "سلام عليكم" (Hello)
   - System classifies as "greeting"
   - Response: "وعليكم السلام يا فندم! كام لتر زيت عندك عشان نجمعهم؟"
   - State updates to: `AWAITING_QUANTITY`

2. User: "عندي 5 لتر" (I have 5 liters)
   - System extracts quantity information
   - Response: "تمام، سجلنا الكمية: 5 لتر. ممكن تقول لي عنوانك الكامل؟"
   - State updates to: `AWAITING_ADDRESS`

3. User: "شارع الهرم" (Haram Street)
   - System records address
   - Response: "حلو، سجلنا العنوان: شارع الهرم. عايز تختار هدية مع الطلب؟"
   - State updates to: `AWAITING_GIFT`

4. User: "كوبون خصم" (Discount coupon)
   - System records gift choice
   - Response confirmation message with order details
   - State updates to: `AWAITING_CONFIRMATION`

5. User: "تمام" (OK)
   - System finalizes order
   - Response: Success message with order summary
   - State resets to: `STARTED`

## Security Considerations
- No authentication mechanism is implemented in this version
- Sessions are maintained in memory and not persisted
- For production use, consider adding proper authentication and moving session storage to a database

## Future Improvements
- Add database persistence for conversations
- Implement user authentication
- Expand the NLP model with more intents
- Add support for order tracking

## License
[Your License Information] 