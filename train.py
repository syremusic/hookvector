import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

with open('data/all_lyrics.json', 'r') as f:
	data = json.load(f)

lyrics = []
labels = []
artist_to_id = {artist: i for i, artist in enumerate(data.keys())}
id_to_artist = {i: artist for artist, i in artist_to_id.items()}

for artist, songs in data.items():
    for song in songs:
        lyrics.append(song)
        labels.append(artist_to_id[artist])


# TODO: not sure wtf happening here
train_lyrics, temp_lyrics, train_labels, temp_labels = train_test_split(
    lyrics, labels, test_size=0.4, random_state=42, stratify=labels
)

val_lyrics, test_lyrics, val_labels, test_labels = train_test_split(
    temp_lyrics, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Training samples: {len(train_lyrics)}")
print(f"Validation samples: {len(val_lyrics)}")
print(f"Test samples: {len(test_lyrics)}")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels, tokenizer, max_len=512):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = str(self.lyrics[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            lyric,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = LyricsDataset(train_lyrics, train_labels, tokenizer)
val_dataset = LyricsDataset(val_lyrics, val_labels, tokenizer)
test_dataset = LyricsDataset(test_lyrics, test_labels, tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(artist_to_id)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(actual_labels, predictions)

epochs = 10
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)
    
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f'Train loss: {train_loss}')
    
    val_accuracy = eval_model(model, val_loader, device)
    print(f'Validation accuracy: {val_accuracy}')
    print()

test_accuracy = eval_model(model, test_loader, device)
print(f'Test accuracy: {test_accuracy}')

def predict_artist(lyrics, model, tokenizer, device, id_to_artist):
    model.eval()
    encoding = tokenizer.encode_plus(
        lyrics,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
        
    return id_to_artist[prediction.item()]

new_lyrics = "i'm in love with the shape of you"
predicted_artist = predict_artist(new_lyrics, model, tokenizer, device, id_to_artist)
print(f"\nThe lyrics '{new_lyrics}' are predicted to be by: {predicted_artist}")
