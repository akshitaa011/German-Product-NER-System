"""
Prediction script for NER with CRF
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF

from utils_bio import extract_aspects_from_bio_tags, save_submission


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NER_CRF_Model(nn.Module):
    """NER model with CRF layer."""
    
    def __init__(self, model_name, num_tags, dropout=0.2):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def decode(self, input_ids, attention_mask):
        """Viterbi decoding."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        mask = attention_mask.bool()
        predictions = self.crf.decode(logits, mask=mask)
        
        return predictions


ALLOWED_ASPECTS = {
    1: {
        "Anzahl_Der_Einheiten", "Besonderheiten", "Bremsscheiben-Aussendurchmesser",
        "Bremsscheibenart", "Einbauposition", "Farbe", "Größe", "Hersteller",
        "Herstellernummer", "Herstellungsland_Und_-Region", "Im_Lieferumfang_Enthalten",
        "Kompatible_Fahrzeug_Marke", "Kompatibles_Fahrzeug_Jahr", "Kompatibles_Fahrzeug_Modell",
        "Material", "Maßeinheit", "Modell", "Oberflächenbeschaffenheit",
        "Oe/Oem_Referenznummer(N)", "Produktart", "Produktlinie", "Stärke", "Technologie", "O"
    },
    2: {
        "Anwendung", "Anzahl_Der_Einheiten", "Besonderheiten", "Breite",
        "Einbauposition", "Größe", "Hersteller", "Herstellernummer",
        "Im_Lieferumfang_Enthalten", "Kompatible_Fahrzeug_Marke", "Kompatibles_Fahrzeug_Jahr",
        "Kompatibles_Fahrzeug_Modell", "Länge", "Maßeinheit", "Menge", "Modell",
        "Oe/Oem_Referenznummer(N)", "Produktart", "SAE_Viskosität", "Zähnezahl", "O"
    }
}


def load_model_and_tokenizer(model_dir):
    """Load trained model and tokenizer."""
    print(f"Loading model from {model_dir}...")
    
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(model_dir, 'tag2id.json'), 'r') as f:
        tag2id = json.load(f)
    with open(os.path.join(model_dir, 'id2tag.json'), 'r') as f:
        id2tag = {int(k): v for k, v in json.load(f).items()}
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = NER_CRF_Model(
        config['model_name'],
        config['num_tags'],
        dropout=0.2
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Model loaded")
    return model, tokenizer, tag2id, id2tag, config['max_length']


def predict_single(title, model, tokenizer, id2tag, tag2id, max_length):
    """Predict aspects for single title."""
    tokens = title.split()
    
    if not tokens:
        return []
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        predictions = model.decode(input_ids, attention_mask)
    
    word_ids = encoding.word_ids(batch_index=0)
    token_predictions = []
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            if idx < len(predictions[0]):
                pred_id = predictions[0][idx]
                token_predictions.append(pred_id)
            previous_word_idx = word_idx
    
    o_tag_id = tag2id.get('O', 0)
    while len(token_predictions) < len(tokens):
        token_predictions.append(o_tag_id)
    
    token_predictions = token_predictions[:len(tokens)]
    
    aspects = extract_aspects_from_bio_tags(tokens, token_predictions, id2tag)
    
    return aspects


def load_data(filepath):
    """Load test data."""
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(
        filepath,
        sep='\t',
        encoding='utf-8',
        keep_default_na=False,
        na_values=None
    )
    
    if 'Record Number' in df.columns:
        df = df.rename(columns={
            'Record Number': 'record_id',
            'Category': 'category_id',
            'Title': 'title'
        })
    
    print(f"Loaded {len(df)} records")
    return df


def make_predictions(df, model, tokenizer, tag2id, id2tag, max_length):
    """Generate predictions for all titles."""
    print("Making predictions...")
    
    all_predictions = []
    filtered_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        record_id = row['record_id']
        category_id = row['category_id']
        title = row['title']
        
        aspects = predict_single(title, model, tokenizer, id2tag, tag2id, max_length)
        
        for aspect_name, aspect_value in aspects:
            if aspect_name == "O":
                continue
            
            if not aspect_value or aspect_value.strip() == "":
                continue
            
            if aspect_name not in ALLOWED_ASPECTS.get(int(category_id), set()):
                filtered_count += 1
                continue
            
            all_predictions.append({
                'record_id': record_id,
                'category_id': category_id,
                'aspect_name': aspect_name.strip(),
                'aspect_value': aspect_value.strip()
            })
    
    print(f"\nGenerated {len(all_predictions)} predictions")
    print(f"Filtered {filtered_count} invalid predictions")
    
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description='Predict with CRF')
    parser.add_argument('--input', type=str, required=True, help='Input TSV file')
    parser.add_argument('--output', type=str, required=True, help='Output submission file')
    parser.add_argument('--model_dir', type=str, default='models/bio_crf', help='Model directory')
    
    args = parser.parse_args()
    
    print("Prediction with BIO + CRF")
    print(f"Model: {args.model_dir}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    model, tokenizer, tag2id, id2tag, max_length = load_model_and_tokenizer(args.model_dir)
    
    df = load_data(args.input)
    
    predictions = make_predictions(df, model, tokenizer, tag2id, id2tag, max_length)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_submission(predictions, args.output)
    
    print("\nPrediction complete")
    print(f"Saved to: {args.output}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()