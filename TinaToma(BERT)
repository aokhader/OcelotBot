# Use Google Colab
# Install necessary libraries
!pip install transformers datasets
!pip install rasa
!pip install scikit-learn

# Step 1: Load and preprocess the dataset from Hugging Face and Kaggle
from datasets import load_dataset

# Load core datasets
human_conversation = load_dataset('projjal1/human-conversation-training-data', split='train')
mental_health = load_dataset('thedevastator/amod-mental-health-counseling-conversations-data', split='train')

# Load supplementary dataset (Chatbot Arena Conversations)
chatbot_arena = load_dataset('lmsys/chatbot_arena_conversations', split='train')

# Optional: Sample only a portion of the supplementary dataset to avoid overpowering the well-being focus
chatbot_arena = chatbot_arena.shuffle(seed=42).select(range(1000))  # Taking a random sample of 1000 examples

# Combine core and supplementary datasets (make sure core datasets are prioritized)
combined_dataset = human_conversation.concatenate(mental_health).concatenate(chatbot_arena)

# Step 2: Fine-tune GPT-2 on the combined dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Preprocess the dataset for GPT-2
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the combined dataset
tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned", 
    per_device_train_batch_size=8,
    num_train_epochs=3, 
    logging_dir="./logs",
    logging_steps=10
)

# Trainer for GPT-2
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_dataset
)

# Fine-tune the model
trainer.train()

# Step 3: Sentiment Analysis with BERT
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model for sentiment analysis
sentiment_tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to classify sentiment using BERT
def sentiment_analysis_bert(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_class = torch.argmax(probs).item()
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return sentiment_labels[sentiment_class]

# Example sentiment analysis using BERT
text = "I'm feeling great today!"
print(f"BERT Sentiment: {sentiment_analysis_bert(text)}")

# Step 4: Fine-tune BERT on Mental Health Dataset (optional)
from transformers import Trainer, TrainingArguments

# Select a small subset of mental health dataset for fine-tuning BERT
subset_mental_health = mental_health.shuffle(seed=42).select(range(1000))

# Preprocess mental health counseling dataset for BERT
def preprocess_bert(examples):
    return sentiment_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the dataset
tokenized_mental_health = subset_mental_health.map(preprocess_bert, batched=True)

# Fine-tune BERT on the selected mental health dataset
training_args_bert = TrainingArguments(
    output_dir="./bert-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./bert_logs",
    logging_steps=10
)

bert_trainer = Trainer(
    model=sentiment_model,
    args=training_args_bert,
    train_dataset=tokenized_mental_health
)

# Fine-tune the BERT sentiment model
bert_trainer.train()

# Step 5: Setup Rasa for Multi-Turn Conversations
# Initialize a new Rasa project (do this once in terminal or Colab shell)
# !rasa init --no-prompt

# Add GPT-2 based response generator into Rasa's actions.py file (custom action)
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ActionGenerateResponse(Action):
    def name(self) -> str:
        return "action_generate_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict) -> List[Dict]:

        # Load pre-trained GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

        # Get the latest user message
        user_message = tracker.latest_message.get('text')

        # Tokenize and generate response using GPT-2
        inputs = tokenizer.encode(user_message, return_tensors='pt')
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

        # Decode the response and send to user
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        dispatcher.utter_message(text=response)

        return []

# Step 6: Evaluation of the model using Scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Example true and predicted labels (replace with actual values)
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")

# Step 7: Deploy the chatbot
