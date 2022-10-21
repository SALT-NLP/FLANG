#-----------------------------------------------------------------------------------------------
processed_input_file = "/data/FinPhraseBank.csv"

model_finetuned = '/path/to/model'
seed_val = 17
batch_size = 32
ratio_of_train = 0.8

model_name = "name-of-model-for-huggingface"
#-----------------------------------------------------------------------------------------------
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from pathlib import Path
import os
parent_path = Path().resolve()

BERT_embeddings_tweets = str(parent_path) + BERT_embeddings_tweets
processed_input_file = str(parent_path) + processed_input_file
# model_finetuned = str(parent_path) + model_finetuned
#-----------------------------------------------------------------------------------------------
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def to_raw(string):
    return fr"{string}"
def explain(text,labels,labels_text ,model,tokenizer):
    from transformers_interpret import SequenceClassificationExplainer
    cls_explainer = SequenceClassificationExplainer( model, tokenizer)
    for i in range(len(text)):
        word_attributions = cls_explainer(text[i])
        print("==============================================================================")
        print(text[i])
        print(tokenizer.tokenize(text[i]),labels_text[i])
        for b in word_attributions:
            print(b)
        # print(word_attributions)
        print(cls_explainer.predicted_class_index)
        # print("actual class = ",labels[i])
        ttt = "=============================================================================="
        html = str(text[i])
        if(cls_explainer.predicted_class_index!=labels[i]):
            with open("bert_explaining_incorrect.html", "a+") as file:
                file.write(ttt)
                file.write("<br>")
                file.write(html)
                file.write("<br>")
            cls_explainer.visualize("bert_explaining_incorrect.html",true_class = labels[i])
        else:
            with open("bert_explaining_correct.html", "a+") as file:
                file.write(ttt)
                file.write("<br>")
                file.write(html)
                file.write("<br>")
            cls_explainer.visualize("bert_explaining_correct.html",true_class = labels[i])


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def evaluate1(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals,outputs

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs.hidden_states[1:]

    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings
df = pd.read_csv(processed_input_file, sep = "\t")
df = df.rename(columns={"News": "text", "Price Sentiment": "label"})


df = df[df.label != 'nocode']
possible_labels = df.label.unique()
print(possible_labels)
label_dict = {}
for i in range(0, len(possible_labels)):
    label_dict[possible_labels[i]] = i
# pickle_off1 = open ('Data/label_dict.txt', "rb")
# label_dict = pickle.load(pickle_off1)
df['label'] = df.label.replace(label_dict)
df['text'].fillna("Empty Tweet", inplace = True)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_finetuned, num_labels=len(possible_labels), output_attentions=False, output_hidden_states=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=(1-ratio_of_train), random_state=17, shuffle=False)
X_train = list(range(0, len(df)))
y_train = np.append(y_train,y_val)
y_train = y_train[:len(X_train)]
df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_val, 'data_type1'] = 'val'
df.loc[X_train[0:len(X_train)-len(X_val)], 'data_type1'] = 'train'
df.loc[X_train, 'data_type'] = 'train'
# df.groupby(['label', 'label', 'data_type1']).count()

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type1=='val'].text.values.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    max_length = 128,
    return_tensors='pt'
)
print(df.head())
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type1=='val'].label.values)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

dataloader_train = DataLoader(dataset_train,sampler=SequentialSampler(dataset_train), batch_size=batch_size)
dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

# ------------------------Remove this block if you don't want to train the model
from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)
for epoch in tqdm(range(1, epochs+1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    torch.save(model.state_dict(), f'finetuned_BERT_epoch1_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# ------------------------


evaluated_results =evaluate1(dataloader_train)
evaluated_results1 = evaluate1(dataloader_validation)

