import json
from transformers import BertForSequenceClassification, BertTokenizerFast, ElectraTokenizerFast, ElectraForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import torch.optim as optim 
import os
import numpy as np
import pandas as pd
from time import time


gpu = input("What gpu do you want to use?: ")
assert gpu in ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
modelToUse = input("What model do you want to use?: ")
assert (modelToUse in ['finbert', 'electra', 'bert', 'finelectra', 'bertflang'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

if modelToUse == 'finbert':
    tokenizer = BertTokenizerFast(vocab_file='/datadrive/FinVocab-Uncased.txt', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'electra':
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator', do_lower_case=True, do_basic_tokenize=True)
elif modelToUse == 'finelectra': 
    tokenizer = ElectraTokenizerFast.from_pretrained('Experiment_model/discriminator_model', do_lower_case=True, do_basic_tokenize=True)
else:
    tokenizer = BertTokenizerFast.from_pretrained('BERT/BERT-FLANG', do_lower_case=True, do_basic_tokenize=True)
with open('FiQA_ABSA_task1/task1_headline_ABSA_train.json', 'rb') as f:
    headlines = json.load(f)
with open('FiQA_ABSA_task1/task1_post_ABSA_train.json', 'rb') as f:
    posts = json.load(f)
maxLength = 0 
for key, value in posts.items():
    sentence = value['sentence']
    for item in value['info']:
        target = item['target']
        tokens = tokenizer(sentence, target)
        maxLength = max(maxLength, len(tokens['input_ids']))

input_ids = []
attention_masks = []
token_type_ids = []
sentiment_scores = []
for key, value in headlines.items():
    sentence = value['sentence']
    for item in value['info']:
        target = item['target']
        tokens = tokenizer(sentence, target, return_tensors='pt', padding='max_length', max_length=maxLength)
        input_ids.append(tokens['input_ids'].squeeze())
        attention_masks.append(tokens['attention_mask'].squeeze())
        token_type_ids.append(tokens['token_type_ids'].squeeze())
        sentiment_scores.append(float(item['sentiment_score']))

for key, value in posts.items():
    sentence = value['sentence']
    for item in value['info']:
        target = item['target']
        tokens = tokenizer(sentence, target, return_tensors='pt', padding='max_length', max_length=maxLength)
        input_ids.append(tokens['input_ids'].squeeze())
        attention_masks.append(tokens['attention_mask'].squeeze())
        token_type_ids.append(tokens['token_type_ids'].squeeze())
        sentiment_scores.append(float(item['sentiment_score']))
sentiment_scores = torch.FloatTensor(sentiment_scores)
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
token_type_ids = torch.stack(token_type_ids)
dataset = TensorDataset(input_ids, attention_masks, token_type_ids, sentiment_scores)
testLength = int(len(dataset)  * 0.2)
valLength = int(len(dataset) * 0.1)
trainLength = len(dataset) - valLength - testLength 
print(trainLength, valLength, testLength)
results = []
SEEDS = [78516, 944601, 5768]#[78516]#[78516, 944601, 5768]
eps = 1e-2
BS = [4]#[32, 16, 8, 4, 2]
count = 0
LR = [1e-5]#[1e-7, 1e-6, 1e-5, 1e-4, 1e-3]#[1e-7, 7e-7, 4e-6, 1e-5, 7e-5, 4e-4, 1e-3]
start = time()
for i, seed in enumerate(SEEDS):
    for j, lr in enumerate(LR):
        for k, bs in enumerate(BS):
            count += 1
            print(f'Experiment {count} of {len(SEEDS) * len(BS) * len(LR)}:')
            torch.manual_seed(seed)
            np.random.seed(seed) 
            if modelToUse == 'finbert':
                model = BertForSequenceClassification.from_pretrained('/datadrive/FinBERT-FinVocab-Uncased', num_labels=1).to(device)
            elif modelToUse == 'bert':
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
            elif modelToUse == 'electra':
                model = ElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator', num_labels=1).to(device)
            elif modelToUse == 'finelectra': 
                model = ElectraForSequenceClassification.from_pretrained('Experiment_model/discriminator_model', num_labels=1).to(device)
            else:
                model = BertForSequenceClassification.from_pretrained('BERT/BERT-FLANG', num_labels=1).to(device)
            train, val, test = torch.utils.data.random_split(dataset=dataset, lengths=[trainLength, valLength, testLength])
            dataloaders_dict = {'train': DataLoader(train, batch_size=bs, shuffle=True), 'val': DataLoader(val, batch_size=bs, shuffle=True),
            'test': DataLoader(test, batch_size=bs, shuffle=True)}
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            num_epochs = 100
            early_stopping = 7
            early_stopping_count = 0
            bestMSE = float('inf')
            bestR2 = float('-inf')
            for epoch in range(num_epochs):
                if (early_stopping_count >= early_stopping):
                    break
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                        early_stopping_count += 1
                    else:
                        model.eval()
                        currMSE = 0
                        currR2 = 0
                    all_sentiment_scores = torch.tensor([]).to(device)
                    all_pred = torch.tensor([]).to(device)
                    for batch in dataloaders_dict[phase]:
                        input_ids, attention_masks, token_type_ids, sentiment_scores = batch
                        input_ids = input_ids.to(device)
                        attention_masks = attention_masks.to(device)
                        token_type_ids = token_type_ids.to(device)
                        sentiment_scores = sentiment_scores.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(input_ids = input_ids, attention_mask = attention_masks, token_type_ids=token_type_ids, labels=sentiment_scores)
                            loss = outputs.loss
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            else:
                                currMSE += loss.item() * input_ids.size(0)
                                all_sentiment_scores = torch.cat([all_sentiment_scores, sentiment_scores], dim = 0)
                                all_pred = torch.cat([all_pred, outputs.logits], dim=0)
                    if phase == 'val':
                        currMSE = currMSE / len(val)
                        currR2 = r2_score(all_sentiment_scores.cpu().detach().numpy(), all_pred.cpu().detach().numpy())
                        if currMSE <= bestMSE - eps:
                            bestMSE = currMSE
                            early_stopping_count = 0
                        if currR2 >= bestR2 + eps:
                            bestR2 = currR2
                            early_stopping_count = 0
                        print("Val MSE: ", currMSE)
                        print("Val R2: ", currR2)
                        print("Early Stopping Count: ", early_stopping_count)
            testMSE = 0
            testR2 = 0  
            all_sentiment_scores = torch.tensor([]).to(device)
            all_pred = torch.tensor([]).to(device) 
            for batch in dataloaders_dict['test']:
                input_ids, attention_masks, token_type_ids, sentiment_scores = batch        
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                sentiment_scores = sentiment_scores.to(device)   
                optimizer.zero_grad()   
                with torch.no_grad():
                    outputs = model(input_ids = input_ids, attention_mask = attention_masks, token_type_ids=token_type_ids, labels=sentiment_scores)
                    loss = outputs.loss
                    testMSE += loss.item() * input_ids.size(0)
                    all_sentiment_scores = torch.cat([all_sentiment_scores, sentiment_scores], dim = 0)
                    all_pred = torch.cat([all_pred, outputs.logits], dim=0)
            testMSE = testMSE / len(test)
            testR2 = r2_score(all_sentiment_scores.cpu().detach().numpy(), all_pred.cpu().detach().numpy())
            results.append([seed, lr, bs, bestMSE, bestR2,  testMSE, testR2])

print(results)
print((time() - start)/60.0)
df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val MSE", "Val R2", "Test MSE", "Test R2"])
df.to_csv(f'GridSearchResults/SentimentClassification_{modelToUse}_multiseed.csv', index=False)

