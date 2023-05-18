import numpy as np
import torch


class PytorchChecker:
    TARGET_LABEL = 'NOM'
    OTHER_LABEL = 'OTR'
    DEVICE_TYPE = 'cuda:0'
    BERT_MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'

    def __init__(self, path_model, path_tokenizer):
        self.path = path_model
        self.path_tokenizer = path_tokenizer
        self.model = None
        self.tokenizer = None
        self.label_dict = {self.OTHER_LABEL: 0, self.TARGET_LABEL: 1}
        self.label_dict_inverse = {v: k for k, v in self.label_dict.items()}
        self.device = torch.device(self.DEVICE_TYPE)

        # load model
        self._make_model()

    def _make_model(self):
        # load pre-trained model
        self.model = torch.load(self.path, map_location=self.DEVICE_TYPE)
        self.model.to(self.device)
        print("Load model")

        # pre-load tokenizer
        self.tokenizer = torch.load(self.path_tokenizer, map_location=self.DEVICE_TYPE)
        print("Load tokenizer")

    def encoder(self, examples):
        encoded_data_val = self.tokenizer.batch_encode_plus(
            examples,
            add_special_tokens=True,
            return_attention_mask=False,
            padding='longest',
            return_tensors='pt'
        )
        data_loader = encoded_data_val['input_ids']
        # print(f"\nEncoding: {data_loader}")

        return data_loader

    def predict(self, examples):
        encoded_samples = self.encoder(examples)

        print(f'\nPrediction of {len(encoded_samples)} examples')
        prediction = list()
        self.model.eval()
        inputs = encoded_samples.to(self.device)
        prediction_samples, true_vals = [], []
        with torch.no_grad():
            outputs = self.model(inputs)
        logits = outputs[0].detach().cpu().numpy()
        prediction_samples.append(logits)
        prediction_list = np.concatenate(prediction_samples, axis=0)
        if np.size(prediction_list, axis=0) == len(examples):
            prediction = list(map(lambda x: np.argmax(x), prediction_list))

        return prediction
