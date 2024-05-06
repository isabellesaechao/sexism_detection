import torch
import pandas as pd
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, f1_score
from tqdm import tqdm

# Dataframes for unlabelled data
preprocessed_gab_data = '../../data/preprocessed_unlabelled_gab.csv'
df1 = pd.read_csv(preprocessed_gab_data)

preprocessed_reddit_data = '../../data/preprocessed_unlabelled_reddit.csv'
df2 = pd.read_csv(preprocessed_reddit_data)

preprocessed_data = '../../data/preprocessed_data_aggregated.csv'
df = pd.read_csv(preprocessed_data)

# Using only 25% of the amount of labelled data from both gab and reddit (.5:1 ratio)
truncate_length = int(0.25 * len(df))
df1_trunc = df1.iloc[:truncate_length]
df2_trunc = df2.iloc[:truncate_length]
df_unlabelled = pd.concat([df1_trunc, df2_trunc], ignore_index=True)

# Move the model to the appropriate device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load the saved model that we trained from D2
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "../D2_models/trained_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)

# Create a function to make predictions on a dataset
def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Working on the batch"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)

            outputs = model(input_ids)

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.detach().cpu().numpy())

    return predictions

# Load the unlabelled data
# unlabelled_train_data = UnlabelledDataset(list(df_unlabelled['cleaned_text'].astype(str)), tokenizer)
# unlabelled_data_loader = DataLoader(unlabelled_train_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
unlabelled_data = df_unlabelled['cleaned_text'].astype(str).tolist()

tokenized_data = tokenizer(unlabelled_data, truncation=True, padding=True)
input_ids = torch.tensor(tokenized_data['input_ids'])
attention_mask = torch.tensor(tokenized_data['attention_mask'])
dataset = TensorDataset(input_ids, attention_mask)

unlabelled_data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Make predictions on the unlabelled data
predictions = predict(model, unlabelled_data_loader)

# #print the results
outputDF = pd.DataFrame()
outputDF['tokenized_text'] = df_unlabelled['tokenized_text']
outputDF['label'] = predictions
outputDF['split'] = 'pseudo'
outputDF.to_csv("../pseudo_data/binary_pseudo_output.csv")

