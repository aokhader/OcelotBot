#This sample Ocelot bot uses gpt-2 as the base model, and the dailydialog dataset from Hugging Face as training data (https://huggingface.co/datasets/roskoN/dailydialog)

# Install necessary libraries
!pip install transformers datasets torch

# Import required libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the Dataset
# Load the dataset from Hugging Face
dataset = load_dataset("roskoN/dailydialog")

# Preview the dataset structure
print(dataset['train'][0])

# Step 2: Tokenization
# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['dialog'], padding="max_length", truncation=True, max_length=128)

# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert to PyTorch tensors
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Step 3: Prepare for Training
# Split into training and validation datasets
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=10
)

# Step 4: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Step 5: Fine-tuning the GPT-2 Model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2_dailydialog")
tokenizer.save_pretrained("./fine_tuned_gpt2_dailydialog")

# Step 6: Inference Function
def generate_response(history, max_length=150):
    input_text = " ".join(history)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example conversation generation
conversation_history = ["Hello! How are you today?", "I'm doing well, thank you! How about you?"]
response = generate_response(conversation_history)
print(response)
