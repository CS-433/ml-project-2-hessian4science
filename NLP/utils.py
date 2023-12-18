from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines
import torch.nn.functional as F
import csv
import torch
import torch.utils.data
from torch import nn, optim
import numpy as np 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup

import re # regex to detect username, url, html entity 
import nltk # to use word tokenize (split the sentence into words)
from nltk.corpus import stopwords # to remove the stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# add rt to remove retweet in dataset (noise)
stop_words.add("rt")



class HATEDataset(Dataset):
    """
    Implement HATEDataset in Pytorch
    """
    def __init__(self, data_repo, tokenizer, sent_max_length=512):
      
        self.tokenizer = tokenizer
        self.max_length = sent_max_length

        self.pad_token = self.tokenizer.pad_token
        self.pad_id = self.tokenizer.pad_token_id

        self.text_samples = []
        self.samples = []
        
        print("Building Hate Dataset...")
        
        with open(data_repo, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)  # or csv.reader(csvfile) if your CSV doesn't have headers
            
            for sample in tqdm(reader):
                self.text_samples.append(sample)

                input_ids = self.tokenizer.encode(sample["tweet"], max_length=sent_max_length, truncation=True, add_special_tokens=False)
                
                label = int(sample["class"])
                self.samples.append({"ids": input_ids, "label": label})

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return deepcopy(self.samples[index])

    def padding(self, inputs, max_length=-1):
        """
        Pad inputs to the max_length.
        
        INPUT: 
          - inputs: input token ids
          - max_length: the maximum length you should add padding to.
          
        OUTPUT: 
          - pad_inputs: token ids padded to `max_length` """

        if max_length < 0:
            max_length = max(list(map(len, inputs)))

        pad_inputs = [ each + [self.pad_id] * (max_length - len(each)) for each in inputs]
        

        return pad_inputs
        
    def collate_fn(self, batch):
        """
        Convert batch inputs to tensor of batch_ids and labels.
        
        INPUT: 
          - batch: batch input, with format List[Dict1{"ids":..., "label":...}, Dict2{...}, ..., DictN{...}]
          
        OUTPUT: 
          - tensor_batch_ids: torch tensor of token ids of a batch, with format Tensor(List[ids1, ids2, ..., idsN])
          - tensor_labels: torch tensor for corresponding labels, with format Tensor(List[label1, label2, ..., labelN])
        """

        batch_ids =  [each["ids"] for each in batch]
        tensor_batch_ids = torch.tensor(self.padding(batch_ids, self.max_length))
        
        batch_labels = [each["label"] for each in batch]
        tensor_labels = torch.tensor(batch_labels).long()
        
        return tensor_batch_ids, tensor_labels
    
    def get_text_sample(self, index):
        return deepcopy(self.text_samples[index])
    
    def decode_class(self, class_ids):
        """
        Decode to output the predicted class name.
        
        INPUT: 
          - class_ids: index of each class.
          
        OUTPUT: 
          - labels_from_ids: a list of label names. """

        print(class_ids)
        label_name_list = [self.id_to_label[each] for each in class_ids]

        
        return label_name_list


def compute_metrics(predictions, gold_labels):
    """
    Compute evaluation metrics (accuracy and F1 score) for NLI task.
    
    INPUT: 
      - gold_labels: real labels;
      - predictions: model predictions.
    OUTPUT: 4 float scores
      - accuracy score (float);
      - f1 score for each class (3 classes in total).
    """
    predictions, gold_labels = torch.tensor(predictions), torch.tensor(gold_labels)
    acc = torch.sum(predictions == gold_labels).item() / len(gold_labels)
    tp = [((predictions == c) & (gold_labels == c)).sum().item() for c in range(3)]
    fp = [((predictions == c) & (gold_labels != c)).sum().item() for c in range(3)]
    fn = [((predictions != c) & (gold_labels == c)).sum().item() for c in range(3)]
    precision = [tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0 for c in range(3)]
    recall = [tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0 for c in range(3)]
    f1 = [2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0 for c in range(3)]

    return acc, f1[0], f1[1], f1[2]






def train(train_dataset:HATEDataset, dev_dataset, model, device, batch_size, epochs,
          learning_rate, warmup_percent, max_grad_norm, count):
    '''
    Train models with predefined datasets.

    INPUT:
      - train_dataset: dataset for training
      - dev_dataset: dataset for evlauation
      - model: model to train
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - epochs: total epochs to train the model
      - learning_rate: learning rate of optimizer
      - warmup_percent: percentage of warmup steps
      - max_grad_norm: maximum gradient for clipping
      - model_save_root: path to save model checkpoints
    '''
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn
    )

    weights = torch.tensor([10, 1.25, 10])

    total_steps =  epochs * batch_size
    warmup_steps = int(warmup_percent * total_steps)
    
    # set up AdamW optimizer and constant learning rate scheduleer with warmup (get_constant_schedule_with_warmup)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    
    
    model.zero_grad()
    best_dev_macro_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        
        train_loss_accum = 0
        epoch_train_step = 0
        predicted = []

        for batch in tqdm(train_dataloader, desc="Training"):
            # Set the gradients of all optimized parameters to zero
            optimizer.zero_grad()

            epoch_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            # get model's single-batch outputs and loss
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits, labels, weight=count)
            
            # conduct back-proporgation
            loss.backward()
        

            # trancate gradient to max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            train_loss_accum += loss.mean().item()

            # step forward optimizer and scheduler
            optimizer.step()
            scheduler.step()

        
        epoch_train_loss = train_loss_accum / epoch_train_step

        # epoch evaluation
        dev_loss, acc, f1_ent, f1_neu, f1_con = evaluate(dev_dataset, model, device, batch_size)
        macro_f1 = (f1_ent + f1_neu + f1_con) / 3
        
        print(f'\nEpoch: {epoch} | Training Loss: {epoch_train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
        print(f'Epoch {epoch} Sequence Validation:')
        print(f'Accuracy: {acc*100:.2f}% | F1: ({f1_ent*100:.2f}%, {f1_neu*100:.2f}%, {f1_con*100:.2f}%) | Macro-F1: {macro_f1*100:.2f}%')
        
        ##############################################################################################################
        # TODO: Update the highest macro_f1. Save best model and tokenizer to <save_repo>.                           # 
        ##############################################################################################################
        # Replace "..." statement with your code
        if macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = macro_f1
            print("Model Saved!")
        
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
            



def evaluate(eval_dataset:HATEDataset, model, device, batch_size, no_labels=False, result_save_file=None):
    '''
    Evaluate the trained model.

    INPUT: 
      - eval_dataset: dataset for evaluation
      - model: trained model
      - device: hardware device for training ('cpu' or 'cuda')
      - batch_size: batch size for load the dataset
      - no_labels: whether the labels should be used as one input to the model
      - result_save_file: path to save the prediction results
    '''
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size,
        collate_fn=eval_dataset.collate_fn
    )
    
    eval_loss_accum = 0
    eval_step = 0
    batch_preds = []
    batch_labels = []
    
    model.eval()
    
    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        
        eval_step += 1
        
        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits, labels)
            logits = outputs.logits

            
            batch_preds.append(logits.detach().cpu().numpy())
            if not no_labels:
                batch_labels.append(labels.detach().cpu().numpy())
                eval_loss_accum += loss.mean().item()

    pred_labels = torch.argmax(torch.tensor(np.concatenate(batch_preds)), dim=1).numpy()

    
    
    if not no_labels:
        eval_loss = eval_loss_accum / eval_step
        gold_labels = list(np.concatenate(batch_labels))
        acc, f1_ent, f1_neu, f1_con = compute_metrics(pred_labels, gold_labels)
        return eval_loss, acc, f1_ent, f1_neu, f1_con
    else:
        return None



# remove html entity:
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text


# change the user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)

    return text

# remove urls
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)

    return text

# remove unnecessary symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text

# remove stopwords
def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)

    return text

## this function in to clean all the dataset by utilizing all the function above
def preprocess(datas):
    clean = []
    # change the @xxx into "user"
    clean = [change_user(text) for text in datas]
    # remove emojis (specifically unicode emojis)
    clean = [remove_entity(text) for text in clean]
    # remove urls
    clean = [remove_url(text) for text in clean]
    # remove trailing stuff
    clean = [remove_noise_symbols(text) for text in clean]
    # remove stopwords
    clean = [remove_stopwords(text) for text in clean]

    return clean


    