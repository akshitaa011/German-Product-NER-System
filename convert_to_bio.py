import pandas as pd


def convert_to_bio(input_file, output_file):
    """
    Convert raw tagged format to BIO format.
    Args:
        input_file: Path to Tagged_Titles_Train.tsv
        output_file: Path to save BIO format data
    """
    print("Converting to BIO Format")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    df = pd.read_csv(
        input_file,
        sep='\t',
        encoding='utf-8',
        keep_default_na=False,
        na_values=None
    )
    
    print(f"Loaded {len(df)} token records")
    
    converted_rows = []
    
    for record_id, group in df.groupby('Record Number'):
        tokens = group['Token'].tolist()
        tags = group['Tag'].tolist()
        category = group['Category'].iloc[0]
        title = group['Title'].iloc[0]
        
        bio_tags = []
        current_aspect = None
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            is_empty = tag == '' or tag.strip() == '' or pd.isna(tag)
            
            if is_empty:
                if current_aspect:
                    bio_tags.append(f'I-{current_aspect}')
                else:
                    bio_tags.append('O')
            elif tag == 'O':
                bio_tags.append('O')
                current_aspect = None
            else:
                if current_aspect == tag:
                    bio_tags.append(f'B-{tag}')
                else:
                    bio_tags.append(f'B-{tag}')
                    current_aspect = tag
        
        for token, bio_tag in zip(tokens, bio_tags):
            converted_rows.append({
                'Record Number': record_id,
                'Category': category,
                'Title': title,
                'Token': token,
                'Tag': bio_tag
            })
    
    converted_df = pd.DataFrame(converted_rows)
    converted_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    
    print(f"Converted {len(converted_df)} token records")
    print(f"Saved to: {output_file}")
    
    tag_counts = converted_df['Tag'].value_counts()
    b_count = sum(1 for tag in converted_df['Tag'] if tag.startswith('B-'))
    i_count = sum(1 for tag in converted_df['Tag'] if tag.startswith('I-'))
    o_count = sum(1 for tag in converted_df['Tag'] if tag == 'O')
    
    print(f"B- tags: {b_count}")
    print(f"I- tags: {i_count}")
    print(f"O tags: {o_count}")


if __name__ == "__main__":
    convert_to_bio(
        input_file='Tagged_Titles_Train.tsv',
        output_file='Tagged_Titles_Train_BIO.tsv'
    )