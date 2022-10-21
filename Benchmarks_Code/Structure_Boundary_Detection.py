# import required libraries

import glob
import json
import torch
import torch.optim as optim
import numpy as np
import os
import re
import bisect
import pandas as pd
from transformers import BertForTokenClassification, BertTokenizerFast, ElectraTokenizerFast, ElectraForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import f1_score


gpu = input("What gpu do you want to use?: ")
assert gpu in ["0", "1", "2", "3"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_finetuned = '/path/to/model'

model_name = "name-of-model-for-huggingface"
tokenizer = AutoTokenizer.from_pretrained(model_finetuned, do_lower_case=False, do_basic_tokenize=True)
addSpecialTokens = True if input('Add Special tokens?: ') == 'Yes' else False

chunkSize = 512
predictionsFilenamesList = glob.glob('finsbd3_train/*finsbd3.json')
tags_dict = {'sentence': 'S', 'list': 'L', "item1" : 'I1', 'item2' : 'I2', 'item3': 'I3', 'item4': 'I4', 'table': 'T', 'figure': 'F', 'page_footer': 'PF', 'page_header': 'PH'}
label_list =  list(tags_dict.values())
label_list = [tag + 'B' for tag in label_list] + [tag + 'E' for tag in label_list] + ['O']
label_list.sort()
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)
all_input_ids = None
all_attention_masks = None
all_tags = None
for file in predictionsFilenamesList:
    if os.path.basename(file) not in fileNames:
        continue
    with open(file) as f:
        data = json.load(f)
    occurrences = [m.start() for m in re.finditer('\n', data['text'])]
    text = data['text'].replace('\n', '')
    batchTokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    tokens = batchTokens[0]
    input_ids = batchTokens['input_ids'].squeeze()
    masks = batchTokens['attention_mask'].squeeze()
    tags = torch.full((len(tokens),), label_to_id['O'], dtype =torch.long)
    for key, value in data.items():
        if key in tags_dict.keys():
            for entry in value:
                updated_start = entry['start'] - bisect.bisect_left(occurrences, entry['start'])
                updated_end = entry['end'] - bisect.bisect_left(occurrences, entry['end'])
                start_ix = tokens.char_to_token(updated_start)
                end_ix = tokens.char_to_token(updated_end - 1)
                if start_ix is not None:
                    tags[start_ix] = label_to_id[tags_dict[key] + 'B']
                if end_ix is not None:
                    tags[end_ix] = label_to_id[tags_dict[key] + 'E']
    if addSpecialTokens:
        curr_input_ids = torch.split(input_ids, chunkSize - 2)
        curr_input_ids = [F.pad(F.pad(id, (1, 0), value=101), (0,1), value=102) for id in curr_input_ids]
        curr_masks = torch.split(masks, chunkSize - 2)
        curr_masks = [F.pad(mask, (1,1), value=1) for mask in curr_masks]
        curr_tags = torch.split(tags, chunkSize - 2)
        curr_tags = [F.pad(tag, (1,1), value=-100) for tag in curr_tags]
    else:
        curr_input_ids = list(torch.split(input_ids, chunkSize))
        curr_masks = list(torch.split(masks, chunkSize))
        curr_tags = list(torch.split(tags, chunkSize))

    curr_input_ids[-1] = F.pad(curr_input_ids[-1], (0, chunkSize - len(curr_input_ids[-1])))
    curr_input_ids = torch.stack(curr_input_ids)
    curr_masks[-1] = F.pad(curr_masks[-1], (0, chunkSize - len(curr_masks[-1])))
    curr_masks = torch.stack(curr_masks)
    curr_tags[-1] = F.pad(curr_tags[-1], (0, chunkSize - len(curr_tags[-1])), value=-100)
    curr_tags = torch.stack(curr_tags)
    print(file)
    print(type(all_input_ids))
    if all_input_ids is None:
        print("here")
        all_input_ids = curr_input_ids
        all_masks = curr_masks
        all_tags = curr_tags
    else:
        all_input_ids = torch.cat((all_input_ids, curr_input_ids), 0)
        all_masks = torch.cat((all_masks, curr_masks), 0)
        all_tags = torch.cat((all_tags, curr_tags), 0)
all_tags = torch.LongTensor(all_tags)
ds = TensorDataset(all_input_ids, all_masks, all_tags)
testLength = int(len(ds)  * 0.2)
valLength = int(len(ds) * 0.1)
trainLength = len(ds) - valLength - testLength
results = []
weights = torch.ones(num_labels).to(device) * 1 / (num_labels - 1)
weights[label_to_id['O']] = 0
print(weights)
print(trainLength, valLength, testLength)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
SEEDS = [ 944601, 5768]
eps = 1e-2
BS = [16, 8, 4, 2]
count = 0
LR = [1e-6, 1e-5, 1e-4, 1e-3]
for i, seed in enumerate(SEEDS):
    for j, lr in enumerate(LR):
        for k, bs in enumerate(BS):
            count += 1
            print(f'Experiment {count} of {len(SEEDS) * len(BS) * len(LR)}:')
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = AutoModelForTokenClassification.from_pretrained(model_finetuned ,num_labels=num_labels).to(device)

            train, val, test = torch.utils.data.random_split(dataset=ds, lengths=[trainLength, valLength, testLength])
            dataloaders_dict = {'train': DataLoader(train, batch_size=bs, shuffle=True), 'val': DataLoader(val, batch_size=bs, shuffle=True),
            'test': DataLoader(test, batch_size=bs, shuffle=True)}
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
                        print("Val Accuracy: ", currAccuracy)
                        print("Val Cross Entropy: ", currCE)
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
df.to_csv(f'SBD_{addSpecialTokens}_{modelToUse}_multiseed.csv', index=False)
