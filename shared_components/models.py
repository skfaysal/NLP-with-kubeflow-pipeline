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

class DistillBERTClass(torch.nn.Module):
    def __init__(self,bert_path):
        super().__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(bert_path)
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