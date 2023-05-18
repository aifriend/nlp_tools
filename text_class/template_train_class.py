import os
import random
import time
from pprint import pprint

from pandarallel import pandarallel
from torch import optim

from preprocess.lib.TextPreprocess import TextPreprocess

os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LIBRARIES_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from pandas import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from preprocess.lib.NlpTool import NlpTool

from common.ClassFile import ClassFile


class GbcNlpService:
    DEVICE_TYPE = 'cuda'
    SOURCE = ''
    DF_CONTENT = 'Content'
    DF_CATEGORY = 'Category'
    _LOAD_MODEL_ROOT = './'
    DATA_DIR = rf''
    MAX_DOC_NUMBER = 9999
    VOCAB_LIB = f'{_LOAD_MODEL_ROOT}.model.vocab'
    LOAD_PATH = f''
    SAVE_PATH = f'{_LOAD_MODEL_ROOT}.model'
    VALIDATION_SIZE = 0.15
    LEARNING_RATE = 1e-6
    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    CONTENT_EXT = "txt"
    ENCODING = 'utf-8'

    # BETO Bert spanish pre-trained
    BERT_MODEL = 'dccuchile/bert-base-spanish-wwm-uncased'

    def __init__(self):
        self.df = None
        self.label_dict = {}
        self.vocab = ''
        self.dataset_train = None
        self.dataset_val = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(self.DEVICE_TYPE)
        self.nlp_service = NlpTool()
        print(f"Device: {self.device}")

    def load_data(self):
        # read dataset from TXT
        cont_list = list()
        label_list = list()

        if not os.path.isdir(self.DATA_DIR):
            raise ValueError(f"Missing data directory at '{self.DATA_DIR}'")

        # read dataset from directory
        doc_class_1_path = os.path.join(self.DATA_DIR, "T1")
        cont_list, label_list = self._loader(
            doc_class_1_path, "T1", cont_list, label_list)

        doc_class_path = os.path.join(self.DATA_DIR, "OT")
        cont_list, label_list = self._loader(
            doc_class_path, "OT", cont_list, label_list)

        # build vocabulary
        try:
            self.vocab = self.nlp_service.load_vocabulary('', f"{self.VOCAB_LIB}")
            if self.vocab is None or not self.vocab:
                raise ValueError("No vocabulary was found")
        except Exception as _:
            raise ValueError("No vocabulary was found")

        # build dataframe
        self.df = pd.DataFrame(
            list(zip(cont_list, label_list)), columns=['Content', 'Category'])
        print(f"\n{self.df.head()}")

        possible_labels = self.df.Category.unique()
        self.label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            self.label_dict[possible_label] = index
        print(f"\nClasses: {self.label_dict}")

        # add label_text by numbers
        self.df['label'] = self.df.Category.replace(self.label_dict)

        # split train, validation and test dataset
        X_train, X_val, y_train, y_val = train_test_split(self.df.index.values,
                                                          self.df.label.values,
                                                          test_size=self.VALIDATION_SIZE,
                                                          random_state=42,
                                                          stratify=self.df.label.values)

        self.df['data_type'] = ['not_set'] * self.df.shape[0]
        self.df.loc[X_train, 'data_type'] = 'train'
        self.df.loc[X_val, 'data_type'] = 'val'
        print(f"\n{self.df.groupby(['Category', 'label', 'data_type']).count()}\n")

        # clean by dictionary tokenizer
        print(f"Clean dataset content for {len(self.df['Content'])} training examples")
        pandarallel.initialize(progress_bar=True)
        steam = False
        is_spanish = False
        greater = 3
        vocab_dist = 2
        self.df["Content"] = self.df["Content"].parallel_apply(
            lambda x: ' '.join(self.nlp_service.filter_word_list(
                x,
                stem=steam,
                is_spanish=is_spanish,
                vocab=self.vocab,
                greater=greater,
                token_filter=False,
                vocab_dist=vocab_dist)))
        print(f"\nCleaned: {self.df.sample(10)}")
        self.df.dropna(
            axis=0,
            how='any',
            thresh=None,
            subset=None,
            inplace=True
        )
        print(f"\nDropped: {self.df.sample(10)}")

    def _loader(self, path, label, c_list, l_list):
        print(f"Loading from {path}")
        doc_list = ClassFile.list_files_like(path, self.CONTENT_EXT)
        for doc in doc_list[:self.MAX_DOC_NUMBER]:
            try:
                page_text = ClassFile.get_text(doc, encoding=self.ENCODING)
                full_text = TextPreprocess.load_document(page_text)
            except Exception as _:
                full_text = ''

            c_list.append(full_text)
            l_list.append(label)
        return c_list, l_list

    def make_model(self, output_size):
        """
        We are treating each text as its unique sequence, so one sequence will be classified to one labels
        "model/beto_pytorch_uncased" is a smaller pre-trained model.
        Using num_labels to indicate the number of output labels.
        We don’t really care about output_attentions.
        We also don’t need output_hidden_states.
        DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset.
        We use RandomSampler for training and SequentialSampler for validation.
        Given the limited memory in my environment, I set batch_size=3.
        """
        if not self.LOAD_PATH:
            print(f"Loading BERT model: {self.BERT_MODEL}")
            self.model = BertForSequenceClassification.from_pretrained(
                self.BERT_MODEL,
                num_labels=output_size,
                output_attentions=False,
                output_hidden_states=False)
        else:
            try:
                self.model = torch.load(rf"{self.LOAD_PATH}.all", map_location=self.device)
                print(f"\nLoading pre-trained model: {self.LOAD_PATH}.all")
            except Exception as _:
                print(f"\nFail Loading from {self.LOAD_PATH} trying load BERT model: {self.BERT_MODEL}")
                self.model = BertForSequenceClassification.from_pretrained(
                    self.BERT_MODEL,
                    num_labels=output_size,
                    output_attentions=False,
                    output_hidden_states=False)

        self.model.to(self.DEVICE_TYPE)

        """
        To construct an optimizer, we have to give it an iterable containing the parameters to optimize.
        Then, we can specify optimizer-specific options such as the learning rate, epsilon, etc.
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters,
                                     lr=self.LEARNING_RATE,
                                     eps=1e-4)

        """
        Create a schedule with a learning rate that decreases linearly from the initial learning rate
        set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to
        the initial learning rate set in the optimizer.
        """
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=output_size * self.NUM_EPOCHS)

    def tokenizer(self):
        """
        Tokenization is a process to take raw texts and split into tokens, which are numeric data to represent words.
        Constructs a BERT tokenizer. Based on WordPiece.
        Instantiate a pre-trained BERT model configuration to encode our data.
        To convert all the titles from text into encoded form, we use a function called batch_encode_plus,
        We will proceed train and validation data separately.
        The 1st parameter inside the above function is the title text.
        add_special_tokens=True means the sequences will be encoded with the special tokens relative to their model.
        When batching sequences together, we set return_attention_mask=True,
        so it will return the attention mask according to the specific tokenizer defined by the max_length attribute.
        We also want to pad all the titles to certain maximum length.
        We actually do not need to set max_length=256, but just to play it safe.
        return_tensors='pt' to return PyTorch.
        And then we need to split the data into input_ids, attention_masks and labels.
        Finally, after we get encoded data set, we can create training data and validation data.
        """
        print(f"\nLoading BERT tokenizer")
        tokenizer = BertTokenizer.from_pretrained(
            self.BERT_MODEL,
            use_fast=False,
            strip_accents=True,
            do_lower_case=True)
        torch.save(tokenizer, self.SAVE_PATH + ".tokenizer")

        return tokenizer

    def encoding(self, tokenizer):
        encoded_data_train = tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'train'].Content.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt'
        )
        pprint(encoded_data_train)
        # seq_len = [len(i.split()) for i in self.df.Content.values]
        # pd.Series(seq_len).plot.hist(bins=30)
        # train_text.tolist(),
        # max_length = 25,
        # pad_to_max_length = True,
        # truncation = True

        # to tensor dataset
        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.df[self.df.data_type == 'train'].label.values)
        self.dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

        # Data Loaders
        data_loader_train = DataLoader(self.dataset_train,
                                       sampler=RandomSampler(self.dataset_train),
                                       batch_size=self.BATCH_SIZE)

        encoded_data_val = tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'val'].Content.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt'
        )
        # pprint(encoded_data_val)

        # to tensor dataset
        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type == 'val'].label.values)
        self.dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

        # Data Loaders
        data_loader_validation = DataLoader(self.dataset_val,
                                            sampler=SequentialSampler(self.dataset_val),
                                            batch_size=self.BATCH_SIZE)

        return data_loader_train, data_loader_validation

    @staticmethod
    def _f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def _accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}'
                  f' -> {round(len(y_preds[y_preds == label]) / len(y_true), 2)}\n')

    def _evaluate(self, data_loader_validation):
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in data_loader_validation:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(data_loader_validation)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train(self, data_loader_train, data_loader_validation):
        tic = time.time()
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        best_acc = 0.0
        for epoch in tqdm(range(1, self.NUM_EPOCHS + 1)):
            # set train mode
            self.model.train()

            loss_train_total = 0

            progress_bar = tqdm(
                data_loader_train, desc='Epoch {:1d}'.format(epoch),
                leave=False, position=0, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }

                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            loss_train_avg = loss_train_total / len(data_loader_train)
            # tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self._evaluate(data_loader_validation)
            # tqdm.write(f'Validation loss: {val_loss}')

            val_f1 = self._f1_score_func(predictions, true_vals)
            if val_f1 > best_acc:
                best_acc = val_f1
                tqdm.write(f'Best model so far: {best_acc}')
                torch.save(self.model.state_dict(), self.SAVE_PATH)
                print(f"Saving trained model to '{self.SAVE_PATH}'...")
                torch.save(self.model, self.SAVE_PATH + ".all")

            if val_loss < 0.10 and val_f1 >= 0.987:
                tqdm.write(f'\nFinal F1 Score: {best_acc}')
                print(f"\nLoading best-trained model: {self.SAVE_PATH}.all")
                break

        toc = time.time()
        time_elapsed = toc - tic
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        print('Best val accuracy: {:4f}'.format(best_acc))

    def predict(self, data_loader_validation):
        self.model.eval()
        _, predictions, true_vals = self._evaluate(data_loader_validation)
        self._accuracy_per_class(predictions, true_vals)


def main():
    text_service = GbcNlpService()

    print(f"\nPre-process data from {GbcNlpService.DATA_DIR}...")
    text_service.load_data()

    print(f"\nTokenizing data...")
    tokenizer = text_service.tokenizer()
    data_loader_train, data_loader_validation = text_service.encoding(tokenizer)

    print(f"\nCreate model from {GbcNlpService.BERT_MODEL}...")
    output_size = len(data_loader_train)
    text_service.make_model(output_size)

    print(f"\nTraining...")
    text_service.train(data_loader_train, data_loader_validation)

    print(f"\nPrediction...")
    text_service.predict(data_loader_validation)


if __name__ == '__main__':
    main()
