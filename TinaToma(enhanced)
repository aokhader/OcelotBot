# Install necessary libraries
!pip install transformers datasets
!pip install rasa
!pip install scikit-learn
!pip install tensorflow  # For BiLSTM

import torch
import tensorflow as tf
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Step 1: Load and preprocess the dataset
human_conversation = load_dataset('projjal1/human-conversation-training-data', split='train')
mental_health = load_dataset('thedevastator/amod-mental-health-counseling-conversations-data', split='train')
chatbot_arena = load_dataset('lmsys/chatbot_arena_conversations', split='train').shuffle(seed=42).select(range(1000))

combined_dataset = human_conversation.concatenate(mental_health).concatenate(chatbot_arena)

# Step 2: GPT-2 Fine-tuning with BiLSTM for Context Tracking
# Define BiLSTM Model
class BiLSTMContextModel(tf.keras.Model):
    def __init__(self, lstm_units):
        super(BiLSTMContextModel, self).__init__()
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.dense = tf.keras.layers.Dense(512, activation='relu')

    def call(self, inputs):
        lstm_output = self.bi_lstm(inputs)
        return self.dense(lstm_output)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Preprocess data for BiLSTM and GPT-2
def preprocess_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# Tokenize dataset
tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)

# Fine-tune GPT-2 using BiLSTM outputs for context
lstm_model = BiLSTMContextModel(lstm_units=256)

# Convert tokenized dataset to TensorFlow dataset
train_dataset = tokenized_dataset.with_format(type="tensorflow")
features = {x: train_dataset[x] for x in tokenizer.model_input_names}

# Apply BiLSTM
lstm_output = lstm_model(features['input_ids'])

# Now pass the LSTM output into GPT-2 model for response generation
outputs = gpt2_model(input_ids=lstm_output)

# Step 3: Sentiment Analysis with BERT
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

# Step 4: Conversational Memory
# Simple memory approach: Maintain conversation history within Rasa or your custom implementation
conversation_history = []

def add_to_conversation_history(user_input, bot_response):
    conversation_history.append({"user": user_input, "bot": bot_response})

# Rasa action for generating response while maintaining context
class ActionGenerateResponseWithMemory(Action):
    def name(self) -> str:
        return "action_generate_response_with_memory"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        # Get latest message
        user_message = tracker.latest_message.get('text')
        
        # Add memory of previous conversation turns
        conversation_input = " ".join([entry['user'] for entry in conversation_history])
        
        # Tokenize and generate response using GPT-2
        inputs = tokenizer.encode(conversation_input + " " + user_message, return_tensors='pt')
        outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
        
        # Decode the response and add it to the memory
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        add_to_conversation_history(user_message, response)
        
        dispatcher.utter_message(text=response)
        return []

# Step 5: Response Variability
# Introduce randomness/variability in responses
def generate_variable_response(user_message):
    inputs = tokenizer.encode(user_message, return_tensors='pt')
    
    # Sample outputs with different temperature values for variability
    outputs = gpt2_model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,       # Enables randomness
        top_k=50,             # Top-k sampling
        temperature=0.7       # Controls the randomness of predictions
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example response variability
user_message = "What restaurants are open on campus?"
response = generate_variable_response(user_message)
print(f"Variable Response: {response}")

# Step 6: Evaluation using Scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

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

# Step 7: Deploy the chatbot via Rasa or any other framework

