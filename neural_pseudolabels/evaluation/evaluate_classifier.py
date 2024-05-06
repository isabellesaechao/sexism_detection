import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

def pad_sequences(sequences, max_length):
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_length:
            padding = torch.zeros(max_length - len(seq), dtype=torch.long)
            padded_seq = torch.cat((seq, padding), dim=0)
        else:
            padded_seq = seq[:max_length]

        padded_sequences.append(padded_seq)

    return padded_sequences

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    max_length = max([len(t) for t in texts])
    padded_texts = pad_sequences(texts, max_length)
    return torch.stack(padded_texts, dim=0), torch.tensor(labels, dtype=torch.long)

# Load the saved model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to the appropriate device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_path = "../D3_models/trained_pseudo_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)

# Load the dataset
data_path = "../../data/preprocessed_data_aggregated.csv"
df = pd.read_csv(data_path)
df['tokenized_text'] = df['cleaned_text'].astype(str).apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))

# Create a function to make predictions on a dataset
def predict(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

    return predictions, true_labels

# Load the dev set
dev_data = TextDataset(df[df['split'] == 'dev']['tokenized_text'].tolist(), df[df['split'] == 'dev']['label'].tolist())
dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Make predictions on the dev set
predictions, true_labels = predict(model, dev_loader)

#print the results
outputDF = pd.DataFrame()
outputDF['rewire_id'] = df[df['split']=='dev']['rewire_id']
outputDF['predictions'] = predictions
outputDF['true_labels'] = true_labels
outputDF.to_csv("../pseudo_outputs/binary_output")

