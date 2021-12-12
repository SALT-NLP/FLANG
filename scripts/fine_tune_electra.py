from transformers import ElectraTokenizerFast,ElectraModel, ElectraConfig
import csv
import numpy as np
#model_name = 'google/electra-large-discriminator'
bsz = 16
model_name='/datadrive/finance/electra/outputs_mask2/discriminator_model'
tokenizer = ElectraTokenizerFast.from_pretrained(model_name)
dic = {"negative": 2, "positive":1, "neutral": 0}

def read_data(split_dir):
    f = open("data/sentiment_data/" + split_dir + ".csv")
    csv_reader = csv.reader(f, delimiter="\t")
    read_top = False
    text = []
    labels = []
    for row in csv_reader:
        if not read_top:
            read_top = True
            continue
        if (row[1].strip()[-1] == "!") or (row[1].strip()[-1] == "?"):
            text.append(row[1].replace("."," ").strip())
        else:
            text.append(row[1].replace("."," ").strip() + " .")
        labels.append(dic[row[2]])
    return text, labels

train_texts, train_labels = read_data("train")
val_texts, val_labels = read_data("validation")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, ElectraForSequenceClassification, AdamW
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = DistilBertForSequenceClassification.from_pretrained(sys.argv[1], num_labels = 3)
model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels = 3)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
test_texts, test_labels = read_data("test")
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = IMDbDataset(test_encodings, test_labels)
optim = AdamW(model.parameters(), lr=2e-5)

for epoch in range(15):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        print("Loss is", epoch, loss)
model.eval()


val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels = labels)
        loss = outputs[0]
        logits = outputs[1]
        new_lab = torch.argmax(logits, dim= 1)
        acc = (new_lab == labels).sum()/new_lab.shape[0]
        print("Validation Loss is", acc.detach().cpu().numpy(), loss.detach().cpu().numpy())


accu = []
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels = labels)
        loss = outputs[0]
        logits = outputs[1]
        new_lab = torch.argmax(logits, dim= 1)
        acc = (new_lab == labels).sum()/new_lab.shape[0]
        accu.append(acc.detach().cpu().numpy())
        print("Test Loss is", acc.detach().cpu().numpy(), loss.detach().cpu().numpy())
print("Final accuracy is ", np.mean(np.array(accu)))

