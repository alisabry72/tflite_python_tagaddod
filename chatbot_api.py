from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
import random
import re

app = Flask(__name__)

# Load the model and required data
def load_model_and_data():
    # Load the trained model
    model = tf.keras.models.load_model("chatbot_model.keras")
    
    # Load vocabulary
    with open("word_index.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)
    
    # Load training data for responses
    with open("training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
      # Try to load metadata for tag ordering
    try:
        with open("model_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            unique_tags = metadata["unique_tags"]
            print("✅ Loaded tag ordering from metadata")
    except:
        # Extract unique tags from training data if metadata not available
        unique_tags = []
        for intent in training_data['intents']:
            unique_tags.append(intent['tag'])
        print("⚠️ Using tag ordering from training data")
    
    
     # Try to load vectorizer config if available
    vectorizer_config = None
    try:
        with open("vectorizer_config.json", "r", encoding="utf-8") as f:
            vectorizer_config = json.load(f)
            print("✅ Loaded vectorizer configuration")
    except:
        print("⚠️ Vectorizer configuration not found, using word_index directly")
    
    # Extract responses
    responses = {}
    fallback_tag = None
    
    for intent in training_data['intents']:
        responses[intent['tag']] = intent['responses']
        if intent['tag'] == 'fallback':
            fallback_tag = 'fallback'
    
    return model, word_index, unique_tags, responses, vectorizer_config, fallback_tag
    

# Load everything at startup
model, word_index, unique_tags, responses, vectorizer_config, fallback_tag = load_model_and_data()

# Enhanced Arabic text preprocessing to match Colab training
def preprocess_arabic_text(text):
    # Convert to lowercase (though less important for Arabic)
    text = text.lower()
    # Remove non-Arabic characters except spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # Normalize Arabic characters
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    # Normalize alef variations
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize yaa variations
    text = re.sub(r'[يى]', 'ي', text)
    # Normalize taa marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize hamza variations
    text = re.sub(r'[ؤئ]', 'ء', text)
    return text

# Create a function to convert text to sequence
def text_to_sequence(text, word_index, max_len=20):
    # Preprocess the text
    text = preprocess_arabic_text(text)
    
    # Convert to sequence
    words = text.split()
    sequence = np.zeros(max_len, dtype=np.int32)
    
    for i, word in enumerate(words):
        if i < max_len:
            # Use 0 for unknown words (0 is reserved for padding)
            sequence[i] = word_index.get(word, 1)  # 1 for OOV token
    
    return sequence

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    data = request.json
    user_input = data.get('text', '')
    
    if not user_input:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess input
    processed_input = preprocess_arabic_text(user_input)
    
    # Convert input to sequence
    sequence = text_to_sequence(processed_input, word_index)
    
    # Make prediction
    prediction = model.predict(np.array([sequence]))
    predicted_tag_index = np.argmax(prediction[0])
    
    # Get the tag and confidence
    predicted_tag = unique_tags[predicted_tag_index]
    confidence = float(prediction[0][predicted_tag_index])
    
    # Get all confidences for debugging
    all_confidences = {tag: float(prediction[0][i]) for i, tag in enumerate(unique_tags)}
    
    # Set confidence threshold
    confidence_threshold = 0.5
    
    # Use fallback if confidence is too low and fallback tag exists
    threshold_applied = False
    if confidence < confidence_threshold and fallback_tag:
        predicted_tag = fallback_tag
        threshold_applied = True
    
    # Get a random response for the predicted tag
    response = random.choice(responses[predicted_tag])
    
    # Return the prediction results with additional info
    return jsonify({
        'tag': predicted_tag,
        'confidence': confidence,
        'response': response,
        'processed_input': processed_input,
        'threshold_applied': threshold_applied,
        'all_confidences': all_confidences
    })
# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'available_tags': unique_tags
    })

# Add CORS support for testing
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)