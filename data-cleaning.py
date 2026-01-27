import pandas as pd
import numpy as np
import json, pathlib, re, random, os

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

    text = re.sub(r'[^a-z0-9``~!@#$%^&*()\-_=+\[\]{}\\|;:"<>,./?\r\n]', '', str(text))
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    return text.lower()

def prepare_conversations():
    df = pd.read_csv(os.path.join(DATASET_PATH, "human_conversation.csv"))
    # ---- Group rows into conversations using "Hi!/Hi./Hi" in human1 ----
    start_re = re.compile(r"^\s*(hi|hello|hey|good\s+morning|good\s+evening)[!.]?\s*$", re.IGNORECASE)


    return df


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


if __name__ == "__main__":
    mental_health_df = prepare_mh()
    conversations_df = prepare_conversations()

