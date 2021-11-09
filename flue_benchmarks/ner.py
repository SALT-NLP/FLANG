# import required libraries

import torch
import torch.optim as optim 
import numpy as np
from transformers import BertForTokenClassification,  BertTokenizerFast, ElectraForTokenClassification, ElectraTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import os
import pandas as pd 

gpu = input("What gpu do you want to use?: ")
assert gpu in ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
modelToUse = input("What model do you want to use?: ")
assert (modelToUse in ['finbert', 'electra', 'bert', 'finelectra', 'bertflang'])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if modelToUse == 'finbert':
    tokenizer = BertTokenizerFast(vocab_file='./finbert-uncased/FinVocab-Uncased.txt', do_lower_case=False, do_basic_tokenize=True)
elif modelToUse == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=False, do_basic_tokenize=True)
elif modelToUse == 'electra':
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator', do_lower_case=False, do_basic_tokenize=True)
elif modelToUse == 'finelectra': 
    tokenizer = ElectraTokenizerFast.from_pretrained('Experiment_model/discriminator_model', do_lower_case=False, do_basic_tokenize=True)
else:
    tokenizer = BertTokenizerFast.from_pretrained('BERT/BERT-FLANG', do_lower_case=False, do_basic_tokenize=True)
labelAllTokens = True if input('Set other tokens in word to current label?: ') == 'Yes' else False

def loadData(train=True):
    if train:
        fileName = 'data/FIN5.txt'
    else:
        fileName = 'data/FIN3.txt'
    with open(fileName, encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]
    sentencesTags = []
    sentenceTags = []
    sentences = []
    sentence = []
    i = 0
    maxLength = 0
    uniqueTags = set()
    while i < len(lines):
        if not lines[i] and sentence and sentenceTags:
            tokens = tokenizer(sentence, is_split_into_words=True)
            if len(tokens['input_ids']) > 512:
                sentence = []
                sentenceTags = []
                continue
            maxLength = max(maxLength, len(tokens['input_ids']))
            sentences.append(sentence)
            sentencesTags.append(sentenceTags)
            uniqueTags = uniqueTags | set(sentenceTags)
            sentence = []
            sentenceTags = []
        elif lines[i] and "DOCSTART" not in lines[i]:
            word, _, _, tag = lines[i].split(" ")
            sentence.append(word)
            sentenceTags.append(tag)
        i+=1
    label_list = list(uniqueTags)
    label_list.sort()
    label_to_id = {l: i for i, l in enumerate(label_list)}
    print(label_to_id)
    num_labels = len(label_list)
    tokenized_inputs = tokenizer(sentences, max_length=maxLength, padding='max_length', is_split_into_words=True, return_tensors='pt')
    input_ids = tokenized_inputs['input_ids']
    attention_masks = tokenized_inputs['attention_mask']
    labels = []
    for i, label in enumerate(sentencesTags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] if labelAllTokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    labels = torch.LongTensor(labels)

    return TensorDataset(input_ids, attention_masks, labels), num_labels, label_to_id

trainDataset, num_labels, label_to_id = loadData(True)
test, _, _  = loadData(False)
valLength = int(len(trainDataset)  * 0.2)
trainLength = len(trainDataset) - valLength
weights = torch.ones(num_labels).to(device) * 1 / (num_labels - 1)
weights[weights.size(0) - 1] = 0
print(weights)
print("length_dataset", trainLength, valLength, len(test))
criterion = torch.nn.CrossEntropyLoss(weight=weights)
results = []
SEEDS = [78516, 944601, 5768]
eps = 1e-2
BS = [16, 8, 4, 2]
count = 0
LR = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
for i, seed in enumerate(SEEDS):
    for j, lr in enumerate(LR):
        for k, bs in enumerate(BS):
            count += 1
            print(f'Experiment {count} of {len(SEEDS) * len(BS) * len(LR)}:')
            torch.manual_seed(seed)
            np.random.seed(seed)
            train, val = torch.utils.data.random_split(dataset=trainDataset, lengths=[trainLength, valLength])
            dataloaders_dict = {'train': DataLoader(train, batch_size=bs, shuffle=True), 'val': DataLoader(val, batch_size=bs, shuffle=True),
            'test': DataLoader(test, batch_size=bs, shuffle=True)}
            if modelToUse == 'finbert':
                model = BertForTokenClassification.from_pretrained('./finbert-uncased/model',num_labels=num_labels).to(device)
            elif modelToUse == 'bert':
                model = BertForTokenClassification.from_pretrained('bert-base-uncased',num_labels=num_labels).to(device)
            elif modelToUse == 'electra': 
                model = ElectraForTokenClassification.from_pretrained('google/electra-large-discriminator', num_labels=num_labels).to(device)
            elif modelToUse == 'finelectra':  
                model = ElectraForTokenClassification.from_pretrained('Experiment_model/discriminator_model', num_labels=num_labels).to(device)
            else:
                model = BertForTokenClassification.from_pretrained('BERT/BERT-FLANG',num_labels=num_labels).to(device)


            optimizer = optim.AdamW(model.parameters(), lr=lr)
            num_epochs = 100
            early_stopping = 7
            early_stopping_count = 0
            bestAccuracy = float('-inf')
            bestF1 = float('-inf')
            bestCE = float('inf')
            for epoch in range(num_epochs):
                if (early_stopping_count >= early_stopping):
                    break
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                        early_stopping_count += 1
                    else:
                        model.eval()
                    currTotal = 0
                    currCorrect = 0
                    currAccuracy = 0
                    currCE = 0
                    actual = np.array([])
                    pred = np.array([])
                    for input_ids, attention_masks, labels in dataloaders_dict[phase]:
                        input_ids = input_ids.to(device)
                        attention_masks = attention_masks.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
                            active_loss = attention_masks.view(-1) == 1
                            logits = outputs.logits
                            active_logits = logits.view(-1, num_labels)
                            active_labels = torch.where(
                                active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
                            )
                            loss = criterion(active_logits, active_labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            else:
                                currPred = outputs.logits.argmax(dim=-1).detach().cpu().clone().numpy()
                                currActual = labels.detach().cpu().clone().numpy()
                                true_predictions = np.concatenate([
                                    [p for (p, l) in zip(pred, gold_label) if l != -100 and l != label_to_id['O']]
                                    for pred, gold_label in zip(currPred, currActual)
                                ])
                                true_labels = np.concatenate([
                                    [l for (p, l) in zip(pred, gold_label) if l != -100 and l != label_to_id['O']]
                                    for pred, gold_label in zip(currPred, currActual)
                                ])
                                currCorrect += np.sum(true_predictions == true_labels)
                                currTotal += len(true_predictions)
                                currCE += loss.item() * input_ids.size(0)
                                actual = np.concatenate([actual, true_labels], axis=0)
                                pred= np.concatenate([pred, true_predictions], axis=0)
                    if phase == 'val':
                        currAccuracy = currCorrect / currTotal
                        currF1 = f1_score(actual, pred, average='weighted')
                        currCE = currCE / len(val)
                        if currCE <= bestCE + eps:
                            bestCE = currCE
                            early_stopping_count = 0
                        if currAccuracy >= bestAccuracy + eps:
                            bestAccuracy = currAccuracy
                            early_stopping_count = 0
                        if currF1 >= bestF1 + eps:
                            bestF1 = currF1
                            early_stopping_count = 0
                        print("Val Cross Entropy: ", currCE)
                        print("Val Accuracy: ", currAccuracy)
                        print("Val F1: ", currF1)
                        print("Early Stopping Count: ", early_stopping_count)
            testAccuracy = 0
            testTotal = 0
            testCorrect = 0
            testCE = 0
            actual = np.array([])
            pred = np.array([])
            for input_ids, attention_masks, labels in dataloaders_dict['test']:
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)   
                optimizer.zero_grad()   
                with torch.no_grad():
                    outputs = model(input_ids = input_ids, attention_mask = attention_masks, labels=labels)
                    active_loss = attention_masks.view(-1) == 1
                    logits = outputs.logits
                    active_logits = logits.view(-1, num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(criterion.ignore_index).type_as(labels)
                    )
                    loss = criterion(active_logits, active_labels)
                    currPred = outputs.logits.argmax(dim=-1).detach().cpu().clone().numpy()
                    currActual = labels.detach().cpu().clone().numpy()
                    true_predictions = np.concatenate([
                        [p for (p, l) in zip(pred, gold_label) if l != -100 and l != label_to_id['O']]
                        for pred, gold_label in zip(currPred, currActual)
                    ])
                    true_labels = np.concatenate([
                        [l for (p, l) in zip(pred, gold_label) if l != -100 and l != label_to_id['O']]
                        for pred, gold_label in zip(currPred, currActual)
                    ])
                    testTotal += len(true_predictions)
                    testCE += loss.item() * input_ids.size(0)
                    testCorrect += np.sum(true_predictions == true_labels)
                    actual = np.concatenate([actual, true_labels], axis=0)
                    pred= np.concatenate([pred, true_predictions], axis=0)
            testAccuracy = testCorrect/testTotal
            testCE = testCE / len(test)
            testF1 = f1_score(actual, pred, average='weighted')
            results.append([seed, lr, bs, bestCE, bestAccuracy,  bestF1, testCE, testAccuracy, testF1])


df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val CE", "Val Accuracy", "Val F1", "Test CE", "Test Accuracy", "Test F1"])       
df.to_csv(f'GridSearchResults/NER_{labelAllTokens}_{modelToUse}_{len(LR)}_multiseed.csv', index=False)