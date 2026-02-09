import jsonlines
import random, os
import pathlib 
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

JSONL_PATH = pathlib.Path(os.getcwd() + "/jsonl-datasets/")

def create_validation_sample(input_files, output_file, n_samples=300):
    all_samples = []
    
    for file_path in input_files:
        with jsonlines.open(file_path) as reader:
            samples = list(reader)
            all_samples.extend(samples)
    
    # Stratified sampling by source
    random.shuffle(all_samples)
    validation_samples = all_samples[:n_samples]
    
    # Create CSV for manual labeling
    import pandas as pd
    df = pd.DataFrame([
        {
            "id": i,
            "context": sample["context"][:200], 
            "response": sample["response"][:200],
            "auto_sentiment": sample["metadata"]["user_sentiment"],
            "manual_sentiment": "",  
            "source": sample["metadata"]["source"]
        }
        for i, sample in enumerate(validation_samples)
    ])
    
    df.to_csv(output_file, index=False)
    print(f"Created validation file: {output_file}")
    print(f"Please manually label the 'manual_sentiment' column with: positive, neutral, or negative")
    
    return validation_samples

def evaluate_auto_labels(validation_file):
    """
    Compare auto-labels vs manual labels.
    """
    df = pd.read_csv(validation_file)
    
    # Remove unlabeled rows
    df = df[df['manual_sentiment'].notna() & (df['manual_sentiment'] != "")]
    
    y_true = df['manual_sentiment'].str.lower()
    y_pred = df['auto_sentiment'].str.lower()
    
    print("Auto-Labeling Accuracy Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    return accuracy

if __name__ == "__main__":
    # run once and manually label
    print("Creating validation sample for manual labeling...")
    create_validation_sample(
        input_files=[
            JSONL_PATH / "mental_health_labeled.jsonl",
            JSONL_PATH / "human_conversations_labeled.jsonl",
            JSONL_PATH / "chatbot_arena_labeled.jsonl"
        ],
        output_file="validation_sentiment.csv",
        n_samples=300
    )

    # After manual labeling, run evaluation
    # print("\nEvaluating auto-labels against manual labels...")
    # evaluate_auto_labels("validation_sentiment.csv")