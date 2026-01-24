import pandas as pd
import numpy as np
import json, pathlib, re, random, os

# Uniform format for all datasets:
# {Instruction: <string>, Context: <string>, Response: <string>, Metadata: {Sentiment: <value>, Source: <value>, ...}}

DATASET_PATH = pathlib.Path(os.getcwd() + "/raw-datasets/")
JSONL_PATH = pathlib.Path(os.getcwd() + "/jsonl-datasets/")

def create_standard_entry(context, response, source, topic="general"):
    """
    Standardizes inputs into a single format:
    {
        Instruction: <string>, 
        Context: <string>, 
        Response: <string>, 
        Metadata: 
            {
                Sentiment: <value>, 
                Source: <value>, 
                ...
            }
    }
    """
    return {
        "instruction": "Respond as an empathetic AI assistant focused on student well-being and academic success.",
        "context": context.strip(),
        "response": response.strip(),
        "metadata": {
            "source": source,
            "topic": topic
        }
    }

def conversation_setup():
    df = pd.read_csv(DATASET_PATH + "human_conversation.csv", sep="#", names=["human1","human2"])

    def norm_cell(x):
        if pd.isna(x): return ""
        s = str(x).strip()
        if len(s) >= 2 and ((s[0]==s[-1]=='"') or (s[0]==s[-1]=="'")):
            s = s[1:-1]
        return s

    df["human1"] = df["human1"].map(norm_cell)
    df["human2"] = df["human2"].map(norm_cell)
    
    # ---- Group rows into conversations using "Hi!/Hi./Hi" in human1 ----
    start_re = re.compile(r"^\s*(hi|hello|hey|good\s+morning|good\s+evening)[!.]?\s*$", re.IGNORECASE)


    return df


if __name__ == "__main__":
    human_convo = conversation_setup()


