"""
Legacy's Mental Health CounselChat - RoBERTa Training Script
Train a RoBERTa model for mental health topic classification

Usage:
    python train_roberta.py --epochs 10 --batch_size 16 --lr 3e-5
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime


# ========================================
# CONFIGURATION
# ========================================

class Config:
    """Training configuration parameters."""
    def __init__(self):
        self.data_path = 'data/counselchat-data.csv'
        self.output_dir = 'models'
        self.model_name = 'roberta-base'
        self.max_length = 512
        self.batch_size = 16
        self.epochs = 10
        self.learning_rate = 3e-5
        self.train_split = 0.8
        self.gradient_accumulation_steps = 2
        self.warmup_steps = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42


# ========================================
# TOPIC GROUPING
# ========================================

TOPIC_GROUPS = {
    'Addiction & Abuse': ['Addiction', 'Substance Abuse', 'Self-harm', 'Eating Disorders'],
    'Anxiety & Stress': ['Anxiety', 'Stress', 'Sleep Improvement', 'Trauma', 'Self-esteem'],
    'Family & Parenting': ['Family Conflict', 'Parenting', 'Children & Adolescents', 'Marriage'],
    'Relationships': ['Relationships', 'Intimacy', 'Relationship Dissolution', 'Social Relationships'],
    'Mental Health Disorders': ['Depression', "Alzheimer's", 'Diagnosis', 'Grief and Loss'],
    'Violence & Safety': ['Domestic Violence', 'Anger Management', 'Military Issues'],
    'Professional & Legal': ['Career Counseling', 'Workplace Relationships', 'Legal & Regulatory', 'Professional Ethics'],
    'Identity & Spirituality': ['Human Sexuality', 'LGBTQ', 'Spirituality'],
    'Behavioral Changes': ['Behavioral Change', 'Counseling Fundamentals']
}


# ========================================
# DATA PREPROCESSING
# ========================================

def parse_topics(x):
    """Parse topics field which may be string or list."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        if ',' in x:
            return [topic.strip() for topic in x.split(',')]
        else:
            return [x.strip()]
    return []


def map_to_group(topics):
    """Map individual topics to broader topic groups."""
    for group, keywords in TOPIC_GROUPS.items():
        if any(topic in keywords for topic in topics):
            return group
    return 'Other'


def load_and_preprocess_data(config):
    """
    Load dataset and preprocess for training.
    
    Returns:
        df: Preprocessed DataFrame
        le: Fitted LabelEncoder
    """
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING")
    print(f"{'='*60}\n")
    
    # Load dataset
    print(f"Loading dataset from: {config.data_path}")
    df = pd.read_csv(config.data_path)
    print(f"Initial dataset shape: {df.shape}")
    
    # Keep necessary columns
    df = df[['questionText', 'topics']].dropna()
    print(f"After dropping NaN: {df.shape}")
    
    # Parse topics
    df['topics'] = df['topics'].apply(parse_topics)
    
    # Map to topic groups
    df['topic_group'] = df['topics'].apply(map_to_group)
    
    # Remove 'Other' category if too small
    other_count = (df['topic_group'] == 'Other').sum()
    if other_count > 0:
        print(f"Removing {other_count} samples in 'Other' category")
        df = df[df['topic_group'] != 'Other']
    
    # Encode labels
    le = LabelEncoder()
    df['topic_group_encoded'] = le.fit_transform(df['topic_group'])
    
    # Display class distribution
    print("\n📊 Class Distribution:")
    print(df['topic_group'].value_counts())
    print(f"\nTotal samples: {len(df)}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}")
    
    return df, le


# ========================================
# DATASET CLASS
# ========================================

class CounselChatDataset(Dataset):
    """PyTorch Dataset for CounselChat data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


# ========================================
# TRAINING FUNCTIONS
# ========================================

def train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, config):
    """Validate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels


def train_model(config):
    """Main training function."""
    
    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True)
    
    # Load and preprocess data
    df, le = load_and_preprocess_data(config)
    
    # Load tokenizer
    print(f"\n{'='*60}")
    print("MODEL SETUP")
    print(f"{'='*60}\n")
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    
    # Create dataset
    print("Creating datasets...")
    dataset = CounselChatDataset(
        df['questionText'],
        df['topic_group_encoded'],
        tokenizer,
        max_length=config.max_length
    )
    
    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Load model
    print(f"Loading model: {config.model_name}")
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(le.classes_)
    )
    model.to(config.device)
    print(f"Device: {config.device}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training history
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'best_accuracy': 0.0,
        'best_epoch': 0
    }
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}\n")
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, config)
        print(f"Average Training Loss: {train_loss:.4f}")
        
        # Validate
        val_accuracy, val_preds, val_labels = validate(model, val_loader, config)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Save best model
        if val_accuracy > history['best_accuracy']:
            history['best_accuracy'] = val_accuracy
            history['best_epoch'] = epoch + 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(config.output_dir, 'rag_model_checkpoint.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer,
                'label_encoder': le,
                'config': vars(config),
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Saved best model checkpoint (accuracy: {val_accuracy:.4f})")
    
    # Final validation with classification report
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}\n")
    
    print(f"Best validation accuracy: {history['best_accuracy']:.4f} (Epoch {history['best_epoch']})")
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(config.output_dir, 'rag_model_checkpoint.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_accuracy, final_preds, final_labels = validate(model, val_loader, config)
    
    print("\n📊 Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=le.classes_))
    
    # Save training history
    history_path = os.path.join(config.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved to: {history_path}")
    
    return model, tokenizer, le, history


# ========================================
# TESTING FUNCTION
# ========================================

def test_model(model, tokenizer, le, config):
    """Test the trained model with sample inputs."""
    
    print(f"\n{'='*60}")
    print("TESTING MODEL")
    print(f"{'='*60}\n")
    
    test_texts = [
        "I feel anxious all the time, and I don't know how to cope.",
        "My partner and I are having relationship problems.",
        "I am struggling with substance abuse and need help quitting.",
        "I want to quit smoking but can't seem to stop.",
        "My self-esteem is so low that I avoid social situations.",
        "I feel hopeless and have no motivation to get out of bed.",
        "My sibling and I are always in conflict, and it's exhausting.",
        "I'm struggling to come to terms with my sexual orientation.",
        "I feel disconnected from my faith and purpose.",
        "I'm experiencing burnout from my high-pressure job.",
        "My workplace has become toxic, and I can't handle it anymore."
    ]
    
    model.eval()
    
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=config.max_length
        ).to(config.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        predicted_topic = le.inverse_transform([predicted_class])[0]
        
        print(f"📝 Input: {text}")
        print(f"🔮 Predicted Topic: {predicted_topic}\n")


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train RoBERTa for mental health topic classification')
    parser.add_argument('--data_path', type=str, default='data/counselchat-data.csv', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--test', action='store_true', help='Test the model after training')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_length = args.max_length
    
    print("="*60)
    print("🧠 LEGACY'S MENTAL HEALTH COUNSELCHAT")
    print("    RoBERTa Topic Classification Training")
    print("="*60)
    
    # Train model
    model, tokenizer, le, history = train_model(config)
    
    # Test model if requested
    if args.test:
        test_model(model, tokenizer, le, config)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config.output_dir}/rag_model_checkpoint.pth")
    print(f"Best validation accuracy: {history['best_accuracy']:.4f}")
    print("\nNext steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Or test with: python train_roberta.py --test")
    print("="*60)


if __name__ == "__main__":
    main()
