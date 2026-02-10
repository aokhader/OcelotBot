import jsonlines
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np
import pandas as pd  
import os, sys
import matplotlib.pyplot as plt
from collections import Counter
import json

random.seed(42)

JSONL_PATH = Path(os.getcwd() + "/jsonl-datasets/")
LORA_PATH = Path(os.getcwd() + "/lora-datasets/")


def format_for_lora_training(input_files, output_file):
    """
    Transform data into LoRA training format with sentiment tags.
    """
    print("-" * 70)
    print("Formatting data with sentiment conditioning.")
    print("-" * 70)
    
    formatted_data = []
    stats = {
        "total": 0,
        "formatted": 0,
        "skipped_empty": 0,
        "by_source": Counter(),
        "by_sentiment": Counter()
    }
    
    for file_path in input_files:
        print(f"\nProcessing: {file_path.name}")
        
        with jsonlines.open(file_path) as reader:
            for item in reader:
                stats["total"] += 1
                
                # Validate required fields
                context = item.get("context", "").strip()
                response = item.get("response", "").strip()
                
                if not context or not response:
                    stats["skipped_empty"] += 1
                    continue
                
                # Extract metadata
                sentiment = item["metadata"]["user_sentiment"].upper()
                source = item["metadata"]["source"]
                instruction = item.get("instruction", 
                                "Respond empathetically to a student, matching the tone and context of the conversation.")
                
                # Update stats
                stats["formatted"] += 1
                stats["by_source"][source] += 1
                stats["by_sentiment"][sentiment] += 1
                
                # Create formatted prompt with sentiment conditioning
                formatted_prompt = f"""[SENTIMENT: {sentiment}]
                    [INSTRUCTION] {instruction}
                    [CONTEXT] {context}
                    [RESPONSE]"""
                                    
                formatted_entry = {
                    "prompt": formatted_prompt,
                    "completion": response,
                    "metadata": {
                        "source": source,
                        "sentiment": sentiment,
                        "topic": item["metadata"].get("topic", "general")
                    }
                }
                
                formatted_data.append(formatted_entry)
    
    # Save formatted data
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(formatted_data)
    
    # Print statistics
    print(f"\n{'-' * 70}")
    print("FORMATTING RESULTS")
    print(f"{'-' * 70}")
    print(f"Total samples processed:  {stats['total']:,}")
    print(f"Successfully formatted:   {stats['formatted']:,}")
    print(f"Skipped (empty):          {stats['skipped_empty']:,}")
    
    print(f"\nDistribution by Source:")
    for source, count in stats['by_source'].most_common():
        pct = (count / stats['formatted']) * 100
        print(f"  {source:25} {count:6,} ({pct:5.1f}%)")
    
    print(f"\nDistribution by Sentiment:")
    for sentiment, count in stats['by_sentiment'].most_common():
        pct = (count / stats['formatted']) * 100
        print(f"  {sentiment:25} {count:6,} ({pct:5.1f}%)")
    
    print(f"\nSaved formatted data to: {output_file}")
    
    return formatted_data, stats


def create_stratified_splits(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create stratified train/val/test splits based on the sources and the sentiments.
    """
    print("\n" + "-" * 70)
    print("Creating stratified train/val/test splits.")
    print("="*70)
    
    with jsonlines.open(input_file) as reader:
        data = list(reader)
    
    print(f"\nTotal samples to split: {len(data):,}")
    
    # Create stratification keys (combination of source + sentiment)
    stratify_keys = [
        f"{item['metadata']['source']}_{item['metadata']['sentiment']}" 
        for item in data
    ]
    print(f"Unique strata: {len(set(stratify_keys))}")
    
    # First split: train vs (val + test)
    train_data, temp_data, train_keys, temp_keys = train_test_split(
        data,
        stratify_keys,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=stratify_keys
    )
    
    # Second split: val vs test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42,
        stratify=temp_keys
    )
    
    # Save splits
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    
    with jsonlines.open(train_file, mode='w') as f:
        f.write_all(train_data)
    with jsonlines.open(val_file, mode='w') as f:
        f.write_all(val_data)
    with jsonlines.open(test_file, mode='w') as f:
        f.write_all(test_data)
    
    # Print split statistics
    print(f"\n{'='*70}")
    print("SPLIT RESULTS")
    print(f"{'='*70}")
    print(f"Train set: {len(train_data):6,} samples ({len(train_data)/len(data)*100:5.1f}%)")
    print(f"Val set:   {len(val_data):6,} samples ({len(val_data)/len(data)*100:5.1f}%)")
    print(f"Test set:  {len(test_data):6,} samples ({len(test_data)/len(data)*100:5.1f}%)")
    
    # Verify stratification worked
    print(f"\n{'='*70}")
    print("STRATIFICATION VERIFICATION")
    print(f"{'='*70}")
    
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        sentiments = [item['metadata']['sentiment'] for item in split_data]
        sources = [item['metadata']['source'] for item in split_data]
        
        sentiment_counts = Counter(sentiments)
        source_counts = Counter(sources)
        
        print(f"\n{split_name} Set:")
        print(f"  Sentiments:")
        for sentiment in sorted(sentiment_counts.keys()):
            count = sentiment_counts[sentiment]
            pct = (count / len(split_data)) * 100
            print(f"    {sentiment:10} {count:5,} ({pct:5.1f}%)")
        
        print(f"  Sources:")
        for source in sorted(source_counts.keys()):
            count = source_counts[source]
            pct = (count / len(split_data)) * 100
            print(f"    {source:25} {count:5,} ({pct:5.1f}%)")
    
    print(f"\n Saved splits to:")
    print(f"   - {train_file}")
    print(f"   - {val_file}")
    print(f"   - {test_file}")
    
    return train_data, val_data, test_data


def analyze_token_lengths(data_file, model_name="mistralai/Mistral-7B-v0.1"):
    """
    Analyze token length distribution to determine optimal max_length for memory.
    """
    print("\n" + "-" * 70)
    print("Analyzing token length distribution.")
    print("-" * 70)
    
    print(f"\nLoading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"\nTokenizing samples from: {data_file.name}")
    
    full_lengths = []
    prompt_lengths = []
    completion_lengths = []
    
    with jsonlines.open(data_file) as reader:
        for i, item in enumerate(reader):
            if (i + 1) % 200 == 0:
                print(f"Processed {i+1} samples...", end='\r')
            
            # Tokenize full sequence (prompt + completion)
            full_text = item["prompt"] + " " + item["completion"]
            full_tokens = tokenizer(full_text, truncation=False, add_special_tokens=True)
            full_lengths.append(len(full_tokens["input_ids"]))
            
            # Tokenize components separately
            prompt_tokens = tokenizer(item["prompt"], truncation=False, add_special_tokens=True)
            completion_tokens = tokenizer(item["completion"], truncation=False, add_special_tokens=False)
            
            prompt_lengths.append(len(prompt_tokens["input_ids"]))
            completion_lengths.append(len(completion_tokens["input_ids"]))
    
    print(f"\n  Processed {len(full_lengths)} samples total")
    
    # Convert to numpy for statistics
    full_lengths = np.array(full_lengths)
    prompt_lengths = np.array(prompt_lengths)
    completion_lengths = np.array(completion_lengths)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("TOKEN LENGTH STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nFull Sequence (Prompt + Completion):")
    print(f"  Mean:            {full_lengths.mean():8.1f} tokens")
    print(f"  Median:          {np.median(full_lengths):8.1f} tokens")
    print(f"  Std deviation:   {full_lengths.std():8.1f} tokens")
    print(f"  Min:             {full_lengths.min():8} tokens")
    print(f"  Max:             {full_lengths.max():8} tokens")
    print(f"  90th percentile: {np.percentile(full_lengths, 90):8.1f} tokens")
    print(f"  95th percentile: {np.percentile(full_lengths, 95):8.1f} tokens")
    print(f"  99th percentile: {np.percentile(full_lengths, 99):8.1f} tokens")
    
    print(f"\nPrompt Only:")
    print(f"  Mean:            {prompt_lengths.mean():8.1f} tokens")
    print(f"  Median:          {np.median(prompt_lengths):8.1f} tokens")
    print(f"  Max:             {prompt_lengths.max():8} tokens")
    
    print(f"\nCompletion Only:")
    print(f"  Mean:            {completion_lengths.mean():8.1f} tokens")
    print(f"  Median:          {np.median(completion_lengths):8.1f} tokens")
    print(f"  Max:             {completion_lengths.max():8} tokens")
    
    # Calculate recommended max_length
    # Use 95th percentile to cover most data, but cap at 2048 for memory
    recommended_max = min(2048, int(np.percentile(full_lengths, 95)))
    
    # Round up to nearest 128 for GPU efficiency
    # GPUs process tensors more efficiently when dimensions are multiples of 128
    recommended_max = ((recommended_max + 127) // 128) * 128
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f" Recommended max_length: {recommended_max} tokens")
    print(f"\nRationale:")
    print(f"  - Covers ~95% of your data (minimal truncation)")
    print(f"  - Rounded to {recommended_max} for GPU efficiency")
    print(f"  - Capped at 2048 to fit in GPU memory")
    
    # Calculate how many samples will be truncated
    exceeds = (full_lengths > recommended_max).sum()
    exceeds_pct = (exceeds / len(full_lengths)) * 100
    
    print(f"\nImpact:")
    print(f"  Samples exceeding {recommended_max} tokens: {exceeds:,} ({exceeds_pct:.2f}%)")
    print(f"  These will be truncated during training")
    
    if exceeds_pct > 10:
        print(f"\n  WARNING: >10% of samples will be truncated.")
        print(f"   Consider increasing max_length or cleaning long samples.")
    
    # Create visualization
    print(f"\n Creating distribution plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Full sequence distribution
    ax1.hist(full_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(recommended_max, color='red', linestyle='--', linewidth=2, 
                label=f'Recommended max ({recommended_max})')
    ax1.axvline(np.median(full_lengths), color='green', linestyle='--', linewidth=2, 
                label=f'Median ({np.median(full_lengths):.0f})')
    ax1.set_xlim(0, min(full_lengths.max(), recommended_max * 2))
    ax1.set_xlabel('Token Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Full Sequence Token Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Prompt vs Completion
    ax2.hist(prompt_lengths, bins=30, alpha=0.6, label='Prompt', edgecolor='black', color='orange')
    ax2.hist(completion_lengths, bins=30, alpha=0.6, label='Completion', edgecolor='black', color='purple')
    ax2.set_xlim(0, min(full_lengths.max(), recommended_max * 2))
    ax2.set_xlabel('Token Length', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prompt vs Completion Token Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = "token_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    
    print(f"Saved plot to: {plot_file}")
    
    return recommended_max, full_lengths


def generate_report(train_file, val_file, test_file, output_file="lora_dataset_report.txt"):
    """
    Generate detailed report about the LoRA training datasets.
    """
    print("\n" + "-" * 70)
    print("Generating comprehensive dataset report.")
    print("-" * 70)
    
    report = []
    report.append("=" * 70)
    report.append("LoRA TRAINING DATASET REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    total_samples = 0
    
    for split_name, file_path in [("TRAIN", train_file), ("VALIDATION", val_file), ("TEST", test_file)]:
        with jsonlines.open(file_path) as reader:
            data = list(reader)
        
        total_samples += len(data)
        
        report.append("=" * 70)
        report.append(f"{split_name} SET")
        report.append("=" * 70)
        report.append(f"Total samples: {len(data):,}")
        report.append("")
        
        # Sentiment distribution
        sentiments = [item['metadata']['sentiment'] for item in data]
        sentiment_counts = Counter(sentiments)
        report.append("Sentiment Distribution:")
        for sentiment in sorted(sentiment_counts.keys()):
            count = sentiment_counts[sentiment]
            pct = (count / len(data)) * 100
            report.append(f"  {sentiment:10} {count:6,} ({pct:5.1f}%)")
        report.append("")
        
        # Source distribution
        sources = [item['metadata']['source'] for item in data]
        source_counts = Counter(sources)
        report.append("Source Distribution:")
        for source in sorted(source_counts.keys()):
            count = source_counts[source]
            pct = (count / len(data)) * 100
            report.append(f"  {source:25} {count:6,} ({pct:5.1f}%)")
        report.append("")
        
        # Topic distribution
        topics = [item['metadata']['topic'] for item in data]
        topic_counts = Counter(topics)
        report.append("Topic Distribution:")
        for topic in sorted(topic_counts.keys()):
            count = topic_counts[topic]
            pct = (count / len(data)) * 100
            report.append(f"  {topic:15} {count:6,} ({pct:5.1f}%)")
        report.append("")
    
    report.append("=" * 70)
    report.append("OVERALL SUMMARY")
    report.append("=" * 70)
    report.append(f"Total samples across all splits: {total_samples:,}")
    report.append("")
    
    # Write report
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n Report saved to: {output_file}")
    print("\nReport preview:")
    print("-" * 70)
    print(report_text)
    
    return report_text


if __name__ == "__main__":
    
    # Define input files
    input_files = [
        JSONL_PATH / "mental_health_labeled.jsonl",
        JSONL_PATH / "human_conversations_labeled.jsonl",
        JSONL_PATH / "chatbot_arena_labeled.jsonl"
    ]
    
    # Check files exist
    for file_path in input_files:
        if file_path.exists():
            print(f"Found: {file_path.name}")
        else:
            print(f"Missing: {file_path.name}")
            print(f"\nError: Please run data-prep.py first to generate labeled datasets.")
            sys.exit(1)
    
    # Step 1: Format data
    formatted_data, format_stats = format_for_lora_training(
        input_files=input_files,
        output_file=LORA_PATH / "lora_formatted.jsonl"
    )
    
    # Step 2: Create splits
    train_data, val_data, test_data = create_stratified_splits(
        input_file=LORA_PATH / "lora_formatted.jsonl",
        output_dir=LORA_PATH
    )
    
    # Step 3: Analyze token lengths
    recommended_max_length, token_lengths = analyze_token_lengths(
        data_file=LORA_PATH / "train.jsonl"
    )
    
    # Step 4: Generate report
    generate_report(
        train_file=LORA_PATH / "train.jsonl",
        val_file=LORA_PATH / "val.jsonl",
        test_file=LORA_PATH / "test.jsonl"
    )
    
    # Save configuration for training script
    config = {
        "max_length": recommended_max_length,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "total_samples": len(formatted_data),
        "base_model": "mistralai/Mistral-7B-v0.1",
        "sentiment_labels": sorted(format_stats["by_sentiment"].keys()),
        "sources": sorted(format_stats["by_source"].keys())
    }
    
    config_file = LORA_PATH / "dataset_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print("LoRA data prep completed.")