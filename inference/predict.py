import torch
import json
import re
import os
import sys
from transformers import DistilBertTokenizer

ROOT_DIR = os.path.abspath(os.path.dirname( __file__ ))
ROOT_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR) ## Do this to import modules outside of current location. or import from root dir
from shared_components import utils,models


class Predictor:
    def __init__(self, config):
        self.max_len = config['max_len']
        self.bert_path = config['model_repo']
        self.model_path = os.path.join(ROOT_DIR,'artifacts',config['trained_model_name'])
        self.label_mapping_path = os.path.join(ROOT_DIR,'artifacts',config['label_mapping_path'])
        
        # Load the model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_path)
        self.model = models.DistillBERTClass(self.bert_path)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.label_mapping = utils.load_json(self.label_mapping_path)



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
        
    def get_label(self,index):
        for label, idx in self.label_mapping.items():
            if idx == index:
                return label

    def predict(self, text):
        # Preprocess and tokenize the text
        text = self.preprocess_text(text)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Get predictions
        with torch.no_grad():
            outputs = self.model(ids, mask)
            probs = torch.softmax(outputs, dim=1)
            predicted_index = torch.argmax(probs, dim=1).cpu().numpy()[0]

        # Convert the index to the label
        predicted_label = self.get_label(predicted_index)
        return predicted_label


if __name__ == '__main__':
    
    config = utils.load_config(os.path.join(ROOT_DIR,'inference','config.yml'))
    
    predictor = Predictor(config)
    prediction = predictor.predict("what is flight fare from nyc to dubai")
    print(prediction)