import csv
import re  # regex to detect username, url, html entity
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from copy import deepcopy
from nltk.corpus import stopwords  # to remove the stopwords
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm

from transformers import AdamW, get_constant_schedule_with_warmup

import nltk  # to use word tokenize (split the sentence into words)

import optimizers

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Set up stopwords and add additional terms
stop_words = set(stopwords.words('english'))
stop_words.add("rt")  # add 'rt' to remove retweet in dataset (noise)






# ========================== Tweet Dataset  ========================== #
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
    macro_f1 = (f1[0] + f1[1] + f1[0]) / 3

    return acc, f1[0], f1[1], f1[2], macro_f1






def train(train_dataset:HATEDataset, dev_dataset, model, device, batch_size, epochs,
          learning_rate, warmup_percent, max_grad_norm, count, optimizer_option="Adam"):
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


    total_steps =  epochs * batch_size
    warmup_steps = int(warmup_percent * total_steps)
    
    # set up AdamW optimizer and constant learning rate scheduleer with warmup (get_constant_schedule_with_warmup)

    # Initial optimizer
    train_loss, train_acc, train_f1_ent, train_f1_neu, train_f1_con, train_macro_f1 = evaluate(train_dataset, model, device, batch_size)

    dev_loss, acc, f1_ent, f1_neu, f1_con, macro_f1 = evaluate(dev_dataset, model, device, batch_size)
    
    print(f'\nEpoch: {0} | Training Loss: {train_loss:.3f} | Validation Loss: {dev_loss:.3f}')
    print(f'Training set  : Accuracy: {train_acc*100:.2f}% | F1: ({train_f1_ent*100:.2f}%, {train_f1_neu*100:.2f}%, {train_f1_con*100:.2f}% | Macro-F1: {train_macro_f1*100:.2f}%')
    print(f'Validation set: Accuracy: {acc*100:.2f}% | F1: ({f1_ent*100:.2f}%, {f1_neu*100:.2f}%, {f1_con*100:.2f}%) | Macro-F1: {macro_f1*100:.2f}%')

    

    optimizer = getattr(optimizers, optimizer_option)(model.parameters(), lr=learning_rate)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)


    criterion = torch.nn.CrossEntropyLoss(weight=count)
    
    model.zero_grad()
    best_dev_macro_f1 = 0
    train_values = [ [] for _ in range(6)]
    dev_values = [ [] for _ in range(6)]

    for epoch in range(epochs):
        model.train()
        
        train_loss_accum = 0
        epoch_train_step = 0
        batch_predicted = []
        batch_labels = []
        count_batch = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            # Set the gradients of all optimized parameters to zero
            optimizer.zero_grad()

            epoch_train_step += 1

            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            if optimizer_option == "SCRN" or optimizer_option == "SCRN_Momentum":
                ind = len(input_ids) // 2
                input_ids = input_ids[:ind]
                data_for_hessian = input_ids[ind:]

                labels = labels[:ind]
                target_for_hessian = labels[ind:]
                optimizer.set_f(model, data_for_hessian, target_for_hessian, F.cross_entropy)
                
            # get model's single-batch outputs and loss
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits, labels, weight=count)
            logits = outputs.logits
            
            # add logits and labels to batch_predicted and batch_labels
            batch_predicted.append(logits.detach().cpu().numpy())
            batch_labels.append(labels.detach().cpu().numpy())

            # conduct back-proporgation
            loss.backward()
        
        



            # trancate gradient to max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


            # general loss and accuracy

            train_loss_accum = +loss.item()* len(input_ids)

            # step forward optimizer and scheduler
            optimizer.step()
            scheduler.step()

            count_batch+=1
            if len(train_dataloader) /10 < count_batch:


                predicted = torch.argmax(torch.tensor(np.concatenate(batch_predicted)), dim=1).numpy()
                gold_labels = list(np.concatenate(batch_labels))

                train_acc, train_f1_ent, train_f1_neu, train_f1_con, train_macro_f1 = compute_metrics(predicted, gold_labels)
                train_loss_accum = train_loss_accum / count_batch


                # epoch evaluation
                dev_loss, acc, f1_ent, f1_neu, f1_con, macro_f1 = evaluate(dev_dataset, model, device, batch_size)
                



                train_values[0].append(train_loss_accum)
                train_values[1].append(train_acc)
                train_values[2].append(train_f1_ent)
                train_values[3].append(train_f1_neu)
                train_values[4].append(train_f1_con)
                train_values[5].append(train_macro_f1)

                dev_values[0].append(dev_loss)
                dev_values[1].append(acc)
                dev_values[2].append(f1_ent)
                dev_values[3].append(f1_neu)
                dev_values[4].append(f1_con)
                dev_values[5].append(macro_f1)
                

                print(f'\nEpoch: {epoch} | Training Loss: {train_loss_accum:.3f} | Validation Loss: {dev_loss:.3f}')
                print(f'Training set  : Accuracy: {train_acc*100:.2f}% | F1: ({train_f1_ent*100:.2f}%, {train_f1_neu*100:.2f}%, {train_f1_con*100:.2f}% | Macro-F1: {train_macro_f1*100:.2f}%')
                print(f'Validation set: Accuracy: {acc*100:.2f}% | F1: ({f1_ent*100:.2f}%, {f1_neu*100:.2f}%, {f1_con*100:.2f}%) | Macro-F1: {macro_f1*100:.2f}%\n')

                batch_predicted = []
                batch_labels = []
                train_loss_accum = 0
                count_batch = 0

        
        
        ##############################################################################################################
        # TODO: Update the highest macro_f1. Save best model and tokenizer to <save_repo>.                           # 
        ##############################################################################################################
        # Replace "..." statement with your code

        ##############################################################################################################
        # TODO: Update the highest macro_f1. Save best model and tokenizer to <save_repo>.                           # 
        ##############################################################################################################
        # Replace "..." statement with your code
        if macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = macro_f1
            print("Model Saved!")
        
    return train_values, dev_values
            


@torch.no_grad()
def evaluate(eval_dataset:HATEDataset, model, device, batch_size, result_save_file=None):
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
    
    for batch in eval_dataloader:
        
        eval_step += 1
        
        with torch.no_grad():
            batch_tuple = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, labels = batch_tuple

            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits, labels)
            logits = outputs.logits

            
            batch_preds.append(logits.detach().cpu().numpy())

            batch_labels.append(labels.detach().cpu().numpy())
            eval_loss_accum += loss.item()

    pred_labels = torch.argmax(torch.tensor(np.concatenate(batch_preds)), dim=1).numpy()

    
    
    eval_loss = eval_loss_accum / eval_step
    gold_labels = list(np.concatenate(batch_labels))
    acc, f1_ent, f1_neu, f1_con, macro_f1 = compute_metrics(pred_labels, gold_labels)
    return eval_loss, acc, f1_ent, f1_neu, f1_con, macro_f1





    