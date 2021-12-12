from transformers import DistilBertTokenizerFast
import csv
import numpy as np
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
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

test_texts, test_labels = read_data("test")

test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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

test_dataset = IMDbDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained(sys.argv[1], num_labels = 3)
model.to(device)
model.eval()


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
        print("Validation Loss is", acc.detach().cpu().numpy(), loss.detach().cpu().numpy())
print("Final accuracy is ", np.mean(np.array(accu)))
