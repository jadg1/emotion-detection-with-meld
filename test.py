import pandas as pd

# Define dataset path
dataset_path = "C:\\Users\\javie\\OneDrive\\Documents\\MELD"  # Use double backslashes

# Load CSV files
train_df = pd.read_csv(f"{dataset_path}/train_sent_emo.csv")
test_df = pd.read_csv(f"{dataset_path}/dev_sent_emo.csv")  # 'dev' is used as test set in MELD

# Print dataset info
print("Training Data:", train_df.head())
print("Test Data:", test_df.head())

import pandas as pd
from transformers import AutoTokenizer

# Load dataset
# Define dataset path
# Meld DATASET uses dialogues from FRIENDS
dataset_path = "C:\\Users\\javie\\OneDrive\\Documents\\MELD"  # Use double backslashes

train_df = pd.read_csv(f"{dataset_path}/train_sent_emo.csv")
test_df = pd.read_csv(f"{dataset_path}/dev_sent_emo.csv")  # 'dev' is used as test set in MELD

# Load tokenizer (using ClinicalBERT) 
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define emotion labels
emotion_labels = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

# Tokenization function
def preprocess(df):
    df["input_ids"] = df["Utterance"].apply(lambda x: tokenizer(x, padding="max_length", truncation=True)["input_ids"])
    df["label"] = df["Emotion"].apply(lambda x: label_map[x])
    return df

# Apply tokenization
train_df = preprocess(train_df)
test_df = preprocess(test_df)

print("Preprocessing Complete!")

##

import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.input_ids = dataframe["input_ids"].tolist()
        self.labels = dataframe["label"].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = EmotionDataset(train_df)
test_dataset = EmotionDataset(test_df)

print("Dataset Ready for Training!")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained model
num_labels = len(emotion_labels)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Start Training
trainer.train()

# Save model & tokenizer after training
model.save_pretrained("C:/Users/javie/OneDrive/Documents/results")
tokenizer.save_pretrained("C:/Users/javie/OneDrive/Documents/results")

print("Model saved successfully!")

# Training takes some time, so we already have saved the resulting model to try out


