import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import re
# Setting up the device for GPU usage
from torch import cuda
from sklearn.model_selection import train_test_split
import time
import json
import re
import os
import yaml
import logging
import sys

ROOT_DIR = os.path.abspath(os.path.dirname( __file__ ))
ROOT_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR) ## Do this to import modules outside of current location. or import from root dir
from shared_components import utils,models




# TextProcessor class
class TextProcessor:
    def __init__(self, config, root_dir, device):
        self.config = config
        self.root_dir = root_dir
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(root_dir, self.config['bert_path']),
            do_lower_case=True
        )
        self.df_train = None
        self.label_mapping = None

    def create_label_mapping(self):
        label_mapping = {label: idx for idx, label in enumerate(self.df_train['label'].unique())}
        self.df_train['target'] = self.df_train['label'].map(label_mapping)
        self.label_mapping = label_mapping

    def filter_classes_with_single_occurrence(self):
        class_distribution = self.df_train['target'].value_counts()
        classes_to_remove = class_distribution[class_distribution == 1].index
        self.df_train = self.df_train[~self.df_train['target'].isin(classes_to_remove)]

    def load_and_prepare_data(self):
        data_path = os.path.join(self.root_dir, self.config['train_data_path'])
        self.df_train = pd.read_csv(data_path)
        self.df_train.columns = ['label', 'query']
        self.create_label_mapping()
        self.filter_classes_with_single_occurrence()

        train_dataset, valid_dataset = train_test_split(
            self.df_train, 
            test_size=1 - self.config['train_size'], 
            random_state=200, 
            stratify=self.df_train['target'],
            shuffle=True
        )
        return train_dataset, valid_dataset

class tweet_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([i for i in text if not i.isdigit()])
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove usernames
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        # Remove extra spaces
        text = " ".join(text.split())
        return text
    
    def __getitem__(self, index):
#         tweet = str(self.data.query[index])
        tweet = str(self.data.iloc[index]['query'])

        tweet = self.preprocess_text(tweet)
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index]['target'], dtype=torch.float)
        }
        
    def __len__(self):
        return len(self.data)

class ModelTrainer:
    def __init__(self, model, train_dl, valid_dl, device, output_path, learning_rate=1e-5):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = device
        self.output_path = output_path
        self.learning_rate = learning_rate

        self.model.to(device)
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, num_epochs):
        start_time = time.time()
        for epoch in range(num_epochs):
            self.model.train()
            for _, data in enumerate(self.train_dl, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)
                
                outputs = self.model(ids, mask).squeeze()
                loss = self.criterion(outputs, targets.long())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            valid_acc = self.eval_fn()
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation F1: {valid_acc:.4f}')
        
        end_time = time.time() - start_time
        print("Training time:", end_time)
        self.save_model()

    def eval_fn(self):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, d in enumerate(self.valid_dl):
                ids = d["ids"].to(self.device, dtype=torch.long)
                mask = d["mask"].to(self.device, dtype=torch.long)
                targets = d["targets"].to(self.device, dtype=torch.long)

                outputs = self.model(ids=ids, mask=mask)
                probs = torch.softmax(outputs, dim=1)
                fin_outputs.extend(torch.argmax(probs, dim=1).cpu().detach().numpy().tolist())
                fin_targets.extend(targets.cpu().detach().numpy().tolist())

        f1 = metrics.f1_score(fin_targets, fin_outputs, average='weighted')
        return f1

    def save_model(self):
        torch.save(self.model.state_dict(), self.output_path)
        logging.info(f"Model saved to {self.output_path}")


if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(ROOT_DIR,'training','training.log'),  # Log filename
                        filemode='w')  # 'w' for overwrite, 'a' for append
    

    
    config = utils.load_config(os.path.join(ROOT_DIR,'training','config.yml'))

    text_processor = TextProcessor(config, ROOT_DIR, device)
    train_dataset, valid_dataset = text_processor.load_and_prepare_data()
    
    # print("FULL Dataset: {}".format(df_train.shape))
    logging.info("\nTRAIN Dataset: {}".format(train_dataset.shape))
    logging.info("Labels in TRAIN Dataset: {}".format(train_dataset['target'].nunique()))
    logging.info("\nVALID Dataset: {}".format(valid_dataset.shape))
    logging.info("Labels in VALID Dataset: {}".format(valid_dataset['target'].nunique()))


    # Create output directory and save the label mapping
    output_dir = os.path.join(ROOT_DIR, config['output_dir'])
    utils.create_directory_if_not_exists(output_dir)
    utils.save_label_mapping(text_processor.label_mapping, output_dir)
    
    ### Create DataLoader for pytorch
    training_set = tweet_Dataset(train_dataset, text_processor.tokenizer, config['max_len'])
    validation_set = tweet_Dataset(valid_dataset, text_processor.tokenizer, config['max_len'])

    logging.info("Training Set Size: {}".format(len(training_set)))
    logging.info("Validation Set Size: {}".format(len(validation_set)))
    
    train_params = {'batch_size': config['batch_size'],
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = {'batch_size': config['batch_size'],
                    'shuffle': True,
                    'num_workers': 0
                    }

    train_dl = DataLoader(training_set, **train_params)
    valid_dl = DataLoader(validation_set, **valid_params)
    
    ## Training and save the trained model
    model = models.DistillBERTClass(os.path.join(ROOT_DIR,config['bert_path']))
    trainer = ModelTrainer(
                        model, 
                        train_dl, 
                        valid_dl, 
                        device, 
                        output_path = os.path.join(ROOT_DIR,config['output_dir'],config['saved_model_name']), 
                        learning_rate = config['learning_rate'])
    
    trainer.fit(num_epochs=1)