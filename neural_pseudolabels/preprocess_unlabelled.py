import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def clean_text(text):
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove user mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Remove hashtags
    text = re.sub(r'#\S+', '', text)

    # Remove special tokens like [USER] and [URL]
    text = re.sub(r'\[(.*?)\]', '', text)

    # Remove special characters, numbers, and punctuations except for periods, apostrophes, and hyphens
    text = re.sub(r'[^A-Za-z .\'-]+', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(input_file, output_file, tokenizer_name):
    df = pd.read_csv(input_file)

    # Clean the text
    df['cleaned_text'] = df['text'].apply(clean_text)


    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    df['tokenized_text'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))

    # only use part of the data
    df_truncated = df.iloc[:200000]

    # Save the preprocessed data
    df_truncated.to_csv(output_file, index=False)

def main():
    input_file = '../data/gab_1M_unlabelled.csv'
    output_file = '../data/preprocessed_unlabelled_gab.csv'
    tokenizer_name = 'bert-base-uncased' # Change this to the desired tokenizer, e.g., 'roberta-base'

    preprocess_data(input_file, output_file, tokenizer_name)

    input_file = '../data/reddit_1M_unlabelled.csv'
    output_file = '../data/preprocessed_unlabelled_reddit.csv'
    preprocess_data(input_file, output_file, tokenizer_name)


if __name__ == '__main__':
    main()
