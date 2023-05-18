import numpy as np
import torch
from transformers import BertForTokenClassification
from transformers import BertTokenizer

from Configuration import Configuration


class NerNameCheckerModel:
    DEVICE_TYPE = 'cpu'

    def __init__(self, model_path):
        self.conf = Configuration()
        self.model = None
        self.model_path = model_path
        self.tokenizer = None
        self.device = torch.device(self.DEVICE_TYPE)

        # load tokenizer
        self.tokenizing()

        # load model
        self.modeling()

    def tokenizing(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.conf.berto_model,
            use_fast=False,
            strip_accents=True,
            do_lower_case=True)

    def encoder(self, _content_values):
        # encoding
        encoded_data_val = self.tokenizer.batch_encode_plus(
            _content_values,
            add_special_tokens=True,
            return_attention_mask=False,
            padding='longest',
            return_tensors='pt'
        )

        # to tensor dataset
        return encoded_data_val['input_ids']

    def modeling(self):
        # load pre-trained model
        self.model = BertForTokenClassification.from_pretrained(
            self.conf.berto_model)

        # load fine-tuned model
        self.model.load_state_dict(
            torch.load(self.model_path,
                       map_location=torch.device(self.DEVICE_TYPE)))

        # send to device
        self.model.to(self.DEVICE_TYPE)

    def _evaluate(self, data_loader):
        self.model.eval()

        predictions, true_vals = [], []

        with torch.no_grad():
            outputs = self.model(data_loader)

        logits = outputs[0].detach().cpu().numpy()
        predictions.append(logits)

        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def predict(self, _content_list):
        data_loader_validation = self.encoder(_content_list)
        prediction = list()
        prediction_list = self._evaluate(data_loader_validation)
        if np.size(prediction_list, axis=0) == len(_content_list):
            prediction = list(map(lambda x: np.argmax(x), prediction_list))

        return prediction
