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
import mlflow
import mlflow.pytorch

def preprocess_text(text):
    """
    Apply various preprocessing steps to the input text.
    - Remove URLs
    - Remove numbers
    - Remove HTML tags
    - Remove usernames
    """
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

def create_label_mapping(dataframe, label_column='label', target_column='target'):
    """
    Create a label mapping for a pandas dataframe.
    
    Args:
    - dataframe (pd.DataFrame): The dataframe containing the labels.
    - label_column (str): The name of the column containing the labels.
    - target_column (str): The name of the new column to be created for encoded labels.

    Returns:
    - pd.DataFrame: The dataframe with the new encoded label column.
    """
    # Create a mapping from unique labels to integers
    label_mapping = {label: idx for idx, label in enumerate(dataframe[label_column].unique())}

    # Apply the mapping to the dataframe
    dataframe[target_column] = dataframe[label_column].map(label_mapping)
    
    return dataframe, label_mapping

def filter_classes_with_single_occurrence(dataframe, target_column='target'):
    """
    Remove classes with only one occurrence from the dataframe.
    
    Args:
    - dataframe (pd.DataFrame): The dataframe to filter.
    - target_column (str): The name of the column containing the target classes.

    Returns:
    - pd.DataFrame: The filtered dataframe.
    - pd.Series: The new class distribution.
    """
    # Checking the class distribution in the dataset
    class_distribution = dataframe[target_column].value_counts()

    # Identifying classes with only one occurrence
    classes_to_remove = class_distribution[class_distribution == 1].index

    # Removing these classes from the dataframe
    filtered_dataframe = dataframe[~dataframe[target_column].isin(classes_to_remove)]

    # Checking the new class distribution
    new_class_distribution = filtered_dataframe[target_column].value_counts()

    return filtered_dataframe, new_class_distribution

class tweet_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
#         tweet = str(self.data.query[index])
        tweet = str(self.data.iloc[index]['query'])

        tweet = preprocess_text(tweet)
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
    
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(BERT_PATH)
        # Freeze DistilBERT parameters
        for param in self.distill_bert.parameters():
            param.requires_grad = False
            
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 18)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d["ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            targets = d["targets"].to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask)

            # Convert outputs to probabilities and get the predicted class
            probs = torch.softmax(outputs, dim=1)
            fin_outputs.extend(torch.argmax(probs, dim=1).cpu().detach().numpy().tolist())

            fin_targets.extend(targets.cpu().detach().numpy().tolist())

        f1 = metrics.f1_score(fin_targets, fin_outputs, average='weighted')
    return f1

def fit(num_epochs, model, loss_fn, opt, train_dl, valid_dl):
    
    for epoch in range(num_epochs):
        model.train()
        for _,data in enumerate(train_dl, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()

        valid_acc = eval_fn(valid_dl, model,device)
        print('Epoch [{}/{}], Train Loss: {:.4f} and Validation f1 {:.4f}'.format(epoch+1, num_epochs, loss.item(),valid_acc))
        
        mlflow.log_metric("train_loss", loss / len(train_dl))
        mlflow.log_metric("validation_f1", valid_acc, step=epoch)

def predict(text, model_path, label_mapping_path, max_len=16):
    # Load label mapping
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # Function to convert index back to label
    def get_label(index):
        for label, idx in label_mapping.items():
            if idx == index:
                return label

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the model
    model = DistillBERTClass()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Preprocess the text
    text = preprocess_text(text)  # Make sure to include the preprocess_text function from your previous script

    # Tokenize the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    # Get predictions
    with torch.no_grad():
        outputs = model(ids, mask)
        probs = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probs, dim=1).cpu().numpy()[0]

    # Convert the index to the label
    predicted_label = get_label(predicted_index)
    return predicted_label


if __name__ == '__main__':
    # Initialize MLflow run
    mlflow.start_run()
    # Defining some key variables that will be used later on in the training
    MAX_LEN = 46
    BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    BERT_PATH = './distilbert_model/'
    MODEL_PATH = "pytorch_model.bin"
    tokenizer = DistilBertTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case=True
    )
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    df_train = pd.read_csv('./data/atis_intents.csv')
    df_train.columns = ['label', 'query']
    # print(df_train.head(2))
    
    ## Label Mapping
    df_train, label_mapping = create_label_mapping(df_train)
    with open('./output/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    # print(label_mapping)
    
    ## Remove classes that contain only one element
    df_train, new_class_distribution = filter_classes_with_single_occurrence(df_train)
    print(new_class_distribution)
    print(df_train.head(2))
    
    ## Create Dataclass
    train_size = 0.80

    # Perform a stratified split to maintain label proportions
    train_dataset, valid_dataset = train_test_split(df_train, test_size=1-train_size, 
                                                    random_state=200, 
                                                    stratify=df_train['target'],
                                                    shuffle=True)

    print("FULL Dataset: {}".format(df_train.shape))
    print("\nTRAIN Dataset: {}".format(train_dataset.shape))
    print("labels in TRAIN Dataset: ", train_dataset['target'].nunique())
    print("\nVALID Dataset: {}".format(valid_dataset.shape))
    print("labels in VALID Dataset: ", valid_dataset['target'].nunique())

    # Assuming the rest of your code for dataset creation remains the same
    training_set = tweet_Dataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = tweet_Dataset(valid_dataset, tokenizer, MAX_LEN)

    print("Training Set Size:", len(training_set))
    print("Validation Set Size:", len(validation_set))

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    train_dl = DataLoader(training_set, **train_params)
    valid_dl = DataLoader(validation_set, **valid_params)

    ### Define Model
    model = DistillBERTClass()
    model.to(device)

    ## Define loss fucntion
    def loss_fn(outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets.long())
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    ## Training
    start = time.time()
    fit(1, model, loss_fn, optimizer, train_dl,valid_dl)
    
    end_time = time.time() - start
    print("time: ",end_time)
    mlflow.log_metric("training_time", end_time)
    
    ## Save model
    mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), './output/distilbert_model.pth')
    
    ### Inference ###
    model_path = './output/distilbert_model.pth'
    label_mapping_path = './output/label_mapping.json'
    text = "which flights go from milwaukee to tampa and stop in nashville"
    prediction = predict(text, model_path, label_mapping_path)
    print("Predicted label:", prediction)
        