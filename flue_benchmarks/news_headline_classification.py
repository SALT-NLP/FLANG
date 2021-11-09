# import required libraries

import pandas as pd
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import os

gpu = input("What gpu do you want to use?: ")
assert gpu in ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
modelToUse = input("What model do you want to use?: ")
assert (modelToUse in ['finbert', 'electra', 'bert', 'finelectra', 'bertflang'])
df = pd.read_excel('CleanGoldHeadlineDataset.xlsx')
headlines = df['News Headline'].to_list()
labels = {}
for column in df.columns:
    if column != "News Headline":
        labels[column] = df[column].to_numpy()
print(labels.keys())
label = input("Which label do you want to train on?: ")
assert(label in labels.keys())
labels = labels[label]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if modelToUse == 'finbert':
    tokenizer = BertTokenizerFast(vocab_file='/datadrive/FinVocab-Uncased.txt', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'electra':
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'finelectra': 
    tokenizer = ElectraTokenizerFast.from_pretrained('/home/ashah482/Desktop/Experiment_model/discriminator_model', do_lower_case=True, do_basic_tokenize=True)
else:
    tokenizer = BertTokenizerFast.from_pretrained('BERT/BERT-FLANG', do_lower_case=True, do_basic_tokenize=True)
maxLength = 0
headlineInputs = []
for i, headline in enumerate(headlines):
    if isinstance(headline, str):
        tokens = tokenizer(headline)['input_ids']
        headlineInputs.append(headline)
        maxLength = max(maxLength, len(tokens))
    else:
        labels = np.delete(labels, i)
tokens = tokenizer(headlineInputs, return_tensors='pt', padding='max_length', max_length=maxLength)
input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']
labels = torch.LongTensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)
testLength = int(len(dataset)  * 0.2)
valLength = int(len(dataset) * 0.1)
trainLength = len(dataset) - valLength - testLength 
results = []
SEEDS = [78516, 944601, 5768]
eps = 1e-2
BS = [32, 16, 8, 4]
LR = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
count = 0
for i, seed in enumerate(SEEDS):
    for j, lr in enumerate(LR):
        for k, bs in enumerate(BS):
            count += 1
            print(f'Experiment {count} of {len(SEEDS) * len(BS) * len(LR)}:')
            torch.manual_seed(seed)
            np.random.seed(seed) 
            if modelToUse == 'finbert':
                model = BertForSequenceClassification.from_pretrained('/datadrive/FinBERT-FinVocab-Uncased', num_labels=2).to(device)
            elif modelToUse == 'bert':
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
            elif modelToUse == 'electra':
                model = ElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator', num_labels=2).to(device)
            elif modelToUse == 'finelectra': 
                model = ElectraForSequenceClassification.from_pretrained('/home/ashah482/Desktop/Experiment_model/discriminator_model', num_labels=2).to(device)
            else:
                model = BertForSequenceClassification.from_pretrained('BERT/BERT-FLANG', num_labels=2).to(device)
            modelToUse = modelToUse + "experiment"
            train, val, test = torch.utils.data.random_split(dataset=dataset, lengths=[trainLength, valLength, testLength])
            dataloaders_dict = {'train': DataLoader(train, batch_size=bs, shuffle=True), 'val': DataLoader(val, batch_size=bs, shuffle=True),
            'test': DataLoader(test, batch_size=bs, shuffle=True)}
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            num_epochs = 100
            early_stopping = 7
            early_stopping_count = 0
            bestCE = float('inf')
            bestAccuracy = float('-inf')
            bestF1 = float('-inf')
            for epoch in range(num_epochs):
                if (early_stopping_count >= early_stopping):
                    break
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                        early_stopping_count += 1
                    else:
                        model.eval()
                    currCE = 0
                    currAccuracy = 0
                    actual = torch.tensor([]).long().to(device)
                    pred = torch.tensor([]).long().to(device)
                    for input_ids, attention_masks, labels in dataloaders_dict[phase]:
                        input_ids = input_ids.to(device)
                        attention_masks = attention_masks.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
                            loss = outputs.loss
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            else:
                                currCE += loss.item() * input_ids.size(0)
                                currAccuracy += torch.sum(torch.max(outputs.logits, 1)[1] == labels).item()
                                actual = torch.cat([actual, labels], dim=0)
                                pred= torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
                    if phase == 'val':
                        currCE = currCE / len(val)
                        currAccuracy = currAccuracy / len(val)
                        currF1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
                        if currCE <= bestCE - eps:
                            bestCE = currCE
                            early_stopping_count = 0
                        if currAccuracy >= bestAccuracy + eps:
                            bestAccuracy = currAccuracy
                            early_stopping_count = 0
                        if currF1 >= bestF1 + eps:
                            bestF1 = currF1
                            early_stopping_count = 0
                        print("Val CE: ", currCE)
                        print("Val Accuracy: ", currAccuracy)
                        print("Val F1: ", currF1)
                        print("Early Stopping Count: ", early_stopping_count)
            testCE = 0
            testAccuracy = 0
            actual = torch.tensor([]).long().to(device)
            pred = torch.tensor([]).long().to(device)
            for input_ids, attention_masks, labels in dataloaders_dict['test']:
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)   
                optimizer.zero_grad()   
                with torch.no_grad():
                    outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
                    loss = outputs.loss
                    testCE += loss.item() * input_ids.size(0)
                    testAccuracy += torch.sum(torch.max(outputs.logits, 1)[1] == labels).item()
                    actual = torch.cat([actual, labels], dim=0)
                    pred = torch.cat([pred, torch.max(outputs.logits, 1)[1]], dim=0)
            testCE = testCE / len(val)
            testAccuracy = testAccuracy/ len(test)
            testF1 = f1_score(actual.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
            results.append([seed, lr, bs, bestCE, bestAccuracy, bestF1, testCE, testAccuracy, testF1])
df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val Cross Entropy", "Val Accuracy", "Val F1 Score", "Test Cross Entropy", "Test Accuracy", "Test F1 Score"])
df.to_csv(f'GridSearchResults/Gold_{label}_{modelToUse}__multiseed.csv', index=False)