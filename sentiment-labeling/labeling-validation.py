from collections import defaultdict
import jsonlines
import random, os
import pathlib 
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

JSONL_PATH = pathlib.Path(os.getcwd().replace("sentiment-labeling", "") + "jsonl-datasets/")

def create_validation_sample(input_files, output_file, n_samples=300):    
    samples_by_source = defaultdict(list)
    
    for file_path in input_files:
        with jsonlines.open(file_path) as reader:
            for sample in reader:
                source = sample["metadata"]["source"]
                samples_by_source[source].append(sample)
    
    # Sample proportionally from each source
    validation_samples = []
    samples_per_source = n_samples // len(samples_by_source)
    
    for source, samples in samples_by_source.items():
        random.shuffle(samples)
        selected = samples[:samples_per_source]
        validation_samples.extend(selected)

    # If needed, add remaining samples randomly
    remaining = n_samples - len(validation_samples)
    if remaining > 0:
        all_remaining = [s for source_samples in samples_by_source.values() 
                        for s in source_samples if s not in validation_samples]
        random.shuffle(all_remaining)
        validation_samples.extend(all_remaining[:remaining])

    # Create CSV for manual labeling
    df = pd.DataFrame([
        {
            "id": i,
            "context": sample["context"], 
            "response": sample["response"],
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
    # Run once and manually label
    # print("Creating validation sample for manual labeling...")
    # create_validation_sample(
    #     input_files=[
    #         JSONL_PATH / "mental_health_labeled.jsonl",
    #         JSONL_PATH / "human_conversations_labeled.jsonl",
    #         JSONL_PATH / "chatbot_arena_labeled.jsonl"
    #     ],
    #     output_file="validation_sentiment.csv",
    #     n_samples=300
    # )

    # After manual labeling, run evaluation
    print("Evaluating auto-labels against manual labels...")
    evaluate_auto_labels("validation_sentiment.csv")