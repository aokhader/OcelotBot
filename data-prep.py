import pandas as pd
import numpy as np
import json, pathlib, re, random, os, jsonlines
from datasets import load_dataset
from transformers import pipeline

random.seed(21)

DATASET_PATH = pathlib.Path(os.getcwd() + "/raw-datasets/")
JSONL_PATH = pathlib.Path(os.getcwd() + "/jsonl-datasets/")

labeler = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def clean_text(x):
    if pd.isna(x): return ""

    # Remove non-ascii characters
    text = x.encode('ascii', 'ignore').decode('ascii')
    # text = re.sub(r'[^a-z0-9``~!@#$%^&*()\-_=+\[\]{}\\|;:"<>,./?\r\n]', '', str(text))

    # Normalize whitespace and remove URLs
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    return text

def sentiment_labels(input_file, output_file):
    labeled_data = []
    
    i = 0
    with jsonlines.open(input_file) as reader:
        for item in reader:
            i += 1
            if i % 200 == 0:
                print(f"Processed {i} samples from {input_file.name}...")

            # Analyze user context sentiment
            user_sentiment = labeler(item["context"][:512])[0]
            
            # Analyze response sentiment
            response_sentiment = labeler(item["response"][:512])[0]
            
            # Add to metadata
            item["metadata"]["user_sentiment"] = user_sentiment["label"]
            item["metadata"]["user_sentiment_score"] = user_sentiment["score"]
            item["metadata"]["response_sentiment"] = response_sentiment["label"]
            item["metadata"]["response_sentiment_score"] = response_sentiment["score"]
            
            labeled_data.append(item)
    
    # Save labeled data
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(labeled_data)
    
    print(f"Labeled {len(labeled_data)} samples \n")
    return labeled_data

def prepare_conversations():
    fp = pathlib.Path(DATASET_PATH) / "human_conversation.csv"
    df = pd.read_csv(fp, delimiter="#", header=0, names=["Human1", "Human2"])
    
    # Group rows into conversations using common greetings
    start_re = re.compile(r"^\s*(hi|hello|hey|good\s+morning|good\s+evening)[!.]?\s*$", re.IGNORECASE)
    processed_data = []
    current_context = []

    for _, row in df.iterrows():
        h1 = str(row['Human1']).strip()
        h2 = str(row['Human2']).strip()

        if start_re.match(h1):
            # New conversations start
            current_context = [] 

        clean_h1 = clean_text(h1)
        clean_h2 = clean_text(h2)

        if clean_h1 == "" or clean_h2 == "":
            continue  

        # Mark the start of a conversation as context if we have a greeting
        conversation = " | ".join(current_context + [clean_h1]) if current_context else "[START]"

        if clean_h1 and clean_h2:
            entry = {
                "instruction": "Respond empathetically to a student, matching the tone and context of the conversation.",
                "context": conversation,
                "response": clean_h2,
                "metadata": {"source": "human_conversation", "topic": "general"}
            }
            processed_data.append(entry)

        current_context.append(f"User: {clean_h1}")
        current_context.append(f"Assistant: {clean_h2}")
        
        # Keep only the last 3 turns to avoid token bloat
        current_context = current_context[-6:]
            
    return pd.DataFrame(processed_data)


def prepare_mh():
    df = pd.read_csv(
        os.path.join(DATASET_PATH, "amod_health.csv"), 
        names=["Context", "Response"],
        skiprows=1,  # Skip header row
    ).dropna(subset=['Response'])

    mh_unified = []
    for _, row in df.iterrows():
        context = clean_text(row['Context'])
        response = clean_text(row['Response'])

        if context == "" or response == "":
            continue 

        if (
            len(context.split()) < 3 or  
            len(response.split()) < 5 or  
            len(response.split()) > 250 or  
            len(context.split()) > 250
        ):
            # Skip entries that are too short or too long to avoid noise and token bloat
            continue

        mh_unified.append({
            "instruction": "Respond empathetically to a student, matching the tone and context of the conversation.",
            "context": context,
            "response": response,
            "metadata": {"source": "mental_health", "topic": "well-being"}
        })

    return pd.DataFrame(mh_unified)


def prepare_chatbot_arena():
    try:
        ds = load_dataset("lmsys/chatbot_arena_conversations", split='train')
        ds = ds.filter(lambda x: x['language'] == "English")
        ds = ds.shuffle(seed=21).select(range(3000))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()
    
    chatbot_unified = []
    for item in ds:
        # Select the winner's conversation as data point
        model_winner = item['winner']
        if model_winner == "model_a":
            conversation = item['conversation_a']
        elif model_winner == "model_b":
            conversation = item['conversation_b']
        else:
            conversation = random.choice([item['conversation_a'], item['conversation_b']])

        history = []
        for i in range(0, len(conversation) - 1, 2):
            user = conversation[i]
            assistant = conversation[i + 1]
            
            user_text = clean_text(user['content'])
            assisstant_text = clean_text(assistant['content'])

            if user_text == "" or assisstant_text == "":
                continue 
            
            # Context = History of previous turns + current user input
            full_context = " | ".join(history + [user_text])
            
            chatbot_unified.append({
                "instruction": "Respond empathetically to a student, matching the tone and context of the conversation.",
                "context": full_context,
                "response": assisstant_text,
                "metadata": {"source": "lmsys_arena", "topic": "general"}
            })
            
            # Update history for the next turn in this same conversation
            history.append(f"User: {user_text}")
            history.append(f"Assistant: {assisstant_text}")
            
            # Keep only the last 3 turns to avoid token bloat
            history = history[-6:] 

    return pd.DataFrame(chatbot_unified)


if __name__ == "__main__":
    print("Preparing datasets...")

    mental_health_df = prepare_mh()
    print(f"Mental Health Dataset prepared with {len(mental_health_df)} entries.")

    conversations_df = prepare_conversations()
    print(f"Human Conversations Dataset prepared with {len(conversations_df)} entries.")

    chatbot_arena_df = prepare_chatbot_arena()
    print(f"Chatbot Arena Dataset prepared with {len(chatbot_arena_df)} entries.")

    # Write cleaned datset to JSONL files
    for df, name in zip([mental_health_df, conversations_df, chatbot_arena_df],
                        ["mental_health.jsonl", "human_conversations.jsonl", "chatbot_arena.jsonl"]):
        output_path = JSONL_PATH / name
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')

    # Add sentiment labels to datasets and save labeled versions
    print("-" * 50)
    print("Adding sentiment labels to datasets...\n")
    sentiment_labels(JSONL_PATH / "mental_health.jsonl", JSONL_PATH / "mental_health_labeled.jsonl")
    sentiment_labels(JSONL_PATH / "human_conversations.jsonl", JSONL_PATH / "human_conversations_labeled.jsonl")
    sentiment_labels(JSONL_PATH / "chatbot_arena.jsonl", JSONL_PATH / "chatbot_arena_labeled.jsonl")



