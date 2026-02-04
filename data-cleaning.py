import pandas as pd
import numpy as np
import json, pathlib, re, random, os
from datasets import load_dataset

random.seed(21)

DATASET_PATH = pathlib.Path(os.getcwd() + "/raw-datasets/")
JSONL_PATH = pathlib.Path(os.getcwd() + "/jsonl-datasets/")

def create_standard_entry(context, response, source, topic="general"):
    # Standardizes inputs into a single format.

    return {
        "instruction": "Respond as an empathetic AI assistant focused on student well-being and academic success.",
        "context": context.strip(),
        "response": response.strip(),
        "metadata": {
            "source": source,
            "topic": topic
        }
    }

def clean_text(x):
    if pd.isna(x): return ""

    # Remove non-ascii characters
    text = x.encode('ascii', 'ignore').decode('ascii')
    # text = re.sub(r'[^a-z0-9``~!@#$%^&*()\-_=+\[\]{}\\|;:"<>,./?\r\n]', '', str(text))

    # Normalize whitespace and remove URLs
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    return text.lower()

def prepare_conversations():
    df = pd.read_csv(os.path.join(DATASET_PATH, "human_conversation.csv"), header=0, names=["Human1", "Human2"])
    
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

        conversation = " | ".join(current_context + [clean_h1])

        if clean_h1 and clean_h2:
            entry = {
                "instruction": "Respond as an conversational AI assistant.",
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
    df = pd.read_csv(os.path.join(DATASET_PATH, "amod_health.csv"), names=["Context", "Response"]).dropna(subset=['Response'])
    mh_unified = []
    for _, row in df.iterrows():
        mh_unified.append({
            "instruction": "Respond as an empathetic student counselor.",
            "context": clean_text(row['Context']),
            "response": clean_text(row['Response']),
            "metadata": {"source": "mental_health", "topic": "well-being"}
        })

    return pd.DataFrame(mh_unified)


def prepare_chatbot_arena():
    try:
        ds = load_dataset("lmsys/chatbot_arena_conversations", split='train')
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
        else:
            conversation = item['conversation_b']

        history = []
        for i in range(len(conversation) - 1, 2):
            user = conversation[i]
            assistant = conversation[i + 1]
            
            user_text = clean_text(user['content'])
            assisstant_text = clean_text(assistant['content'])
            
            # Context = History of previous turns + current user input
            full_context = " | ".join(history + [user_text])
            
            chatbot_unified.append({
                "instruction": "Respond as an empathetic AI assistant.",
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
    mental_health_df = prepare_mh()
    conversations_df = prepare_conversations()
    chatbot_arena_df = prepare_chatbot_arena()



