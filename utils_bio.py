import pandas as pd
import re
import csv


def clean_value(text):
    """
    Only normalizes whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def is_valid_span(value):
    """
    Check if extracted span is valid.
    Accepts single characters.
    """
    value = value.strip()
    if len(value) == 0:
        return False
    return True


def extract_aspects_from_bio_tags(tokens, tag_ids, id2tag):
    """
    Extract aspect name-value pairs from BIO tagged sequence.
    
    Args:
        tokens: List of word tokens
        tag_ids: List of tag IDs or tag strings
        id2tag: Mapping from tag ID to tag string
    
    Returns:
        List of (aspect_name, aspect_value) tuples
    """
    aspects = []
    current_tokens = []
    current_aspect = None
    
    for token, tag_id in zip(tokens, tag_ids):
        if isinstance(tag_id, int):
            tag = id2tag.get(tag_id, 'O')
        else:
            tag = str(tag_id)
        
        if tag == 'O':
            if current_aspect and current_tokens:
                value = clean_value(" ".join(current_tokens))
                if is_valid_span(value):
                    aspects.append((current_aspect, value))
            current_tokens = []
            current_aspect = None
        
        elif tag.startswith('B-'):
            if current_aspect and current_tokens:
                value = clean_value(" ".join(current_tokens))
                if is_valid_span(value):
                    aspects.append((current_aspect, value))
            
            aspect_name = tag[2:]
            current_aspect = aspect_name
            current_tokens = [token]
        
        elif tag.startswith('I-'):
            aspect_name = tag[2:]
            
            if current_aspect == aspect_name:
                current_tokens.append(token)
            else:
                if current_aspect and current_tokens:
                    value = clean_value(" ".join(current_tokens))
                    if is_valid_span(value):
                        aspects.append((current_aspect, value))
                
                current_aspect = aspect_name
                current_tokens = [token]
        
        else:
            if current_aspect and current_tokens:
                value = clean_value(" ".join(current_tokens))
                if is_valid_span(value):
                    aspects.append((current_aspect, value))
            current_tokens = []
            current_aspect = None
    
    if current_aspect and current_tokens:
        value = clean_value(" ".join(current_tokens))
        if is_valid_span(value):
            aspects.append((current_aspect, value))
    
    return aspects


def load_training_data(filepath):
    """Load BIO format training data."""
    print(f"Loading training data from {filepath}...")
    
    df = pd.read_csv(
        filepath,
        sep='\t',
        encoding='utf-8',
        keep_default_na=False,
        na_values=None
    )
    
    print(f"Loaded {len(df)} token-level records")
    return df


def prepare_ner_dataset(df):
    """Convert token-level data to example format."""
    print("Preparing NER dataset...")
    
    examples = []
    
    if 'Record Number' in df.columns:
        df = df.rename(columns={
            'Record Number': 'record_id',
            'Category': 'category_id',
            'Title': 'title',
            'Token': 'token',
            'Tag': 'tag'
        })
    
    for record_id, group in df.groupby('record_id', sort=False):
        category_id = group.iloc[0]['category_id']
        title = group.iloc[0]['title']
        tokens = group['token'].tolist()
        tags = group['tag'].tolist()
        
        examples.append({
            'record_id': record_id,
            'category_id': category_id,
            'title': title,
            'tokens': tokens,
            'tags': tags
        })
    
    print(f"Prepared {len(examples)} examples")
    return examples


def create_tag_mappings(tags_list):
    """Create bidirectional tag-ID mappings."""
    tag2id = {tag: idx for idx, tag in enumerate(tags_list)}
    id2tag = {idx: tag for idx, tag in enumerate(tags_list)}
    return tag2id, id2tag


def get_unique_tags(examples):
    """Extract unique BIO tags from examples."""
    all_tags = set()
    for example in examples:
        all_tags.update(example['tags'])
    
    def sort_key(tag):
        if tag == 'O':
            return (0, tag)
        elif tag.startswith('B-'):
            return (1, tag)
        elif tag.startswith('I-'):
            return (2, tag)
        else:
            return (3, tag)
    
    tags_list = sorted(list(all_tags), key=sort_key)
    
    print(f"Found {len(tags_list)} unique BIO tags")
    return tags_list


def save_submission(predictions, output_path):
    """
    Save predictions in TSV format.
    TAB-separated, no header, UTF-8 encoding.
    """
    if len(predictions) == 0:
        print("Warning: No predictions")
        df = pd.DataFrame(columns=['record_id', 'category_id', 'aspect_name', 'aspect_value'])
    else:
        df = pd.DataFrame(predictions)
        df = df[['record_id', 'category_id', 'aspect_name', 'aspect_value']]
    
    df.to_csv(
        output_path,
        sep='\t',
        index=False,
        header=False,
        encoding='utf-8',
        quoting=csv.QUOTE_NONE,
        escapechar=None
    )
    
    print(f"Saved {len(df)} predictions to {output_path}")