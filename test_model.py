import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Since the training takes some time, we are going to load the resulting model
# Load the trained model from the correct directory
MODEL_PATH = "C:/Users/javie/OneDrive/Documents/results"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = logits.argmax().item()

    # Emotion labels in MELD dataset
    # We have 6 emotions other than neutral, 7 total
    emotion_labels = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
    return emotion_labels[predicted_class]

# Test with sample sentences
# We have some examples to try out how punctuations and other expressions may affect
test_sentences = [
    "I feel my arm hurt.",
    "My brother fell down the stairs",
    "Why dont you get what I am explaining!",
    "Why dont you get what I am explaining",
    "Mom is now in heaven",
    "Mom is now in hell",
    "Fuck, my arm hurts",
    "HAHA i fell down the stairs",
    "HAHA feliponcio fell down the stairs"
]
#
# Iterate through each sentence and print the sentence before predicting
for sentence in test_sentences:
    print(f"Sentence: {sentence}")  # Print the sentence
    print(f"Predicted Emotion: {predict_emotion(sentence)}\n")  # Print the prediction with spacing


##Weight

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# send text to online component
# take into account audio to text
# audio online transmission velocity
# size down to raspberry pi requirements\


