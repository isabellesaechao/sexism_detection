import torch
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Load the preprocessed data
preprocessed_data = '../../data/preprocessed_data_aggregated.csv'
df = pd.read_csv(preprocessed_data)

#concatenate the pseudo data with the data
pseudo_data = "../pseudo_data/11way_pseudo_output.csv"
df2 = pd.read_csv(pseudo_data)
df_pseudo = df2[['tokenized_text', 'label', 'split']]
df_train = df[df['split'] == 'train'][['tokenized_text', 'label', 'split']]
merged_df = pd.concat([df_pseudo, df_train])

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length=512):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_text = ast.literal_eval(self.texts[idx])
        # Pad the tokenized text with zeros to match the max_length
        padded_text = tokenized_text + [0] * (self.max_length - len(tokenized_text))
        return torch.tensor(padded_text), torch.tensor(self.labels[idx])

# Create a new DataFrame with only sexist posts
df_sexist = df[df['label_sexist'] == 'sexist'].copy()

# Create a new column with the updated 11-way classification labels
label_map = {
    '1.1 threats of harm': 0,
    '1.2 incitement and encouragement of harm': 1,
    '2.1 descriptive attacks': 2,
    '2.2 aggressive and emotive attacks': 3,
    '2.3 dehumanising attacks & overt sexual objectification': 4,
    '3.1 casual use of gendered slurs, profanities, and insults': 5,
    '3.2 immutable gender differences and gender stereotypes': 6,
    '3.3 backhanded gendered compliments': 7,
    '3.4 condescending explanations or unwelcome advice': 8,
    '4.1 supporting mistreatment of individual women': 9,
    '4.2 supporting systemic discrimination against women as a group': 10
    }
df_sexist['new_label'] = df_sexist['label_vector'].apply(lambda x: label_map[x])



# Create PyTorch datasets and data loaders
train_data = TextDataset(merged_df['tokenized_text'].tolist(), merged_df['label'].tolist())
val_data = TextDataset(df_sexist[df_sexist['split'] == 'dev']['tokenized_text'].tolist(), df_sexist[df_sexist['split'] == 'dev']['new_label'].tolist())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=11)


# Move the model to the appropriate device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using GPU for training.")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU for training.")
else:
    device = torch.device('cpu')
    print("Using CPU for training.")

model.to(device)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

# Fine-tune the model
best_val_loss = float('inf')
patience = 3
curr_patience = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    # Wrap the training loop with tqdm for a progress bar
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()

        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # check for early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      curr_patience = 0
    else:
      curr_patience += 1
    if curr_patience >= patience:
      break
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the model and tokenizer
output_dir = '../D3_models/trained_pseudo_model_11way'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
