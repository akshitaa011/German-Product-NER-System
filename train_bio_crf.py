"""
Training script for XLM-RoBERTa + CRF model
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torchcrf import CRF
from tqdm import tqdm
import argparse

from utils_bio import (
    load_training_data, prepare_ner_dataset,
    create_tag_mappings, get_unique_tags
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NERDataset(Dataset):
    """Dataset class for NER with BIO tagging."""
    
    def __init__(self, examples, tokenizer, tag2id, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example['tokens']
        tags = example['tags']
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                tag = tags[word_idx] if word_idx < len(tags) else 'O'
                label_ids.append(self.tag2id.get(tag, self.tag2id['O']))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }


class NER_CRF_Model(nn.Module):
    """NER model with CRF layer for sequence labeling."""
    
    def __init__(self, model_name, num_tags, dropout=0.2):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            mask = (labels != -100)
            mask[:, 0] = True
            
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            
            loss = -self.crf(logits, labels_crf, mask=mask, reduction='mean')
            return loss, logits
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=mask)
            return logits, predictions
    
    def decode(self, input_ids, attention_mask):
        """Viterbi decoding for prediction."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        mask = attention_mask.bool()
        predictions = self.crf.decode(logits, mask=mask)
        
        return predictions


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss, _ = model(input_ids, attention_mask, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            predictions = model.decode(input_ids, attention_mask)
            
            for pred_seq, label_seq, attn_mask in zip(predictions, labels, attention_mask):
                valid_length = attn_mask.sum().item()
                pred_tensor = torch.tensor(pred_seq[:valid_length], device=device)
                label_tensor = label_seq[:valid_length]
                
                mask = label_tensor != -100
                if mask.sum() > 0:
                    correct += ((pred_tensor == label_tensor) & mask).sum().item()
                    total += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train NER with CRF')
    parser.add_argument('--train_data', type=str, default='Tagged_Titles_Train_BIO.tsv')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--output_dir', type=str, default='models/bio_crf')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    
    print("Training NER with CRF")
    print(f"Model: {args.model_name}")
    print(f"Device: {DEVICE}")
    
    df = load_training_data(args.train_data)
    examples = prepare_ner_dataset(df)
    
    tags_list = get_unique_tags(examples)
    tag2id, id2tag = create_tag_mappings(tags_list)
    
    print(f"Unique tags: {len(tags_list)}")
    
    train_examples, val_examples = train_test_split(
        examples, test_size=0.15, random_state=42
    )
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = NERDataset(train_examples, tokenizer, tag2id, args.max_length)
    val_dataset = NERDataset(val_examples, tokenizer, tag2id, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print("Initializing model...")
    model = NER_CRF_Model(args.model_name, len(tags_list), dropout=0.2)
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print("Starting training")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            print("New best model")
            best_val_loss = val_loss
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
            tokenizer.save_pretrained(args.output_dir)
            
            with open(os.path.join(args.output_dir, 'tag2id.json'), 'w') as f:
                json.dump(tag2id, f)
            with open(os.path.join(args.output_dir, 'id2tag.json'), 'w') as f:
                json.dump(id2tag, f)
            
            config = {
                'model_name': args.model_name,
                'num_tags': len(tags_list),
                'max_length': args.max_length
            }
            with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
    
    print("\nTraining complete")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()