"""
Extract test data from listing dataset
Records 100,001 to 125,000
"""

import pandas as pd
import argparse


def extract_test_data(input_file, output_file, start_record=100001, end_record=125000):
    """
    Extract test records from listing data.
    
    Args:
        input_file: Path to Listing_Titles.tsv
        output_file: Path to save test data
        start_record: First record number
        end_record: Last record number
    """
    print("Extracting test data")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Range: {start_record} - {end_record}")
    
    df = pd.read_csv(
        input_file,
        sep='\t',
        encoding='utf-8',
        keep_default_na=False,
        na_values=None
    )
    
    print(f"Total records: {len(df)}")
    
    test_df = df[(df['Record Number'] >= start_record) & (df['Record Number'] <= end_record)]
    
    print(f"Extracted {len(test_df)} test records")
    
    test_df.to_csv(
        output_file,
        sep='\t',
        index=False,
        encoding='utf-8'
    )
    
    print(f"Saved to: {output_file}")
    print(f"Categories: {test_df['Category'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description='Extract test data')
    parser.add_argument('--input', type=str, default='Listing_Titles.tsv')
    parser.add_argument('--output', type=str, default='test_data.tsv')
    parser.add_argument('--start', type=int, default=100001)
    parser.add_argument('--end', type=int, default=125000)
    
    args = parser.parse_args()
    
    extract_test_data(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()