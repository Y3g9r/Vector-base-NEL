import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertModel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import datetime as dt
import gc

class DisambiguationDataset(Dataset):
    def __init__(self, samples,labels):
        self.samples = samples
        self.labels = labels
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        items = {"text_input_ids": torch.tensor(self.samples[index][0]),
                 "text_input_mask": torch.tensor(self.samples[index][1]),
                 "text_segment_ids": torch.tensor(self.samples[index][2]),
                 "def_input_ids": torch.tensor(self.samples[index][3]),
                 "def_input_mask": torch.tensor(self.samples[index][4]),
                 "def_segment_ids": torch.tensor(self.samples[index][5]),
                 "label": torch.tensor(self.labels[index])}
        return items


class NerualNet(nn.Module):
    def __init__(self, hidden_size=768, max_seq_len=388):
        super(NerualNet, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False

        self.text_pooling = nn.MaxPool1d(kernel_size=max_seq_len, stride=1)
        self.def_pooling = nn.MaxPool1d(kernel_size=max_seq_len, stride=1)

        self.cos = torch.nn.CosineSimilarity()

    def forward(self, input_ids, input_mask, segment_ids):
        bert_output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(bert_output[1].squeeze()))
            else:
                h += self.fc(dropout(bert_output[1].squeeze()))
        return h / len(self.dropouts)

    def get_defenition_embedding(self, input_ids_def, token_type_ids_def, attention_mask_def):
        # with torch.no_grad():
        #     _, pooled_output, _ = self.model(tokens_tensor, segments_tensors)
        #     return pooled_output[0]
        with torch.no_grad():
            output = self.model(input_ids=input_ids_def, token_type_ids=token_type_ids_def,
                                attention_mask=attention_mask_def)
            hidden_states = output[2]
        # from [# layers, # batches, # tokens, # features] to [# tokens, # layers, # features]
        token_dim = torch.stack(hidden_states, dim=0)
        token_dim = torch.squeeze(token_dim, dim=1)
        token_dim = token_dim.permute(0, 1, 2)
        cat_vec = torch.cat(((token_dim[-4][0] + token_dim[-3][0] + token_dim[-2][0] + token_dim[-1][0]),), dim=0)
        return cat_vec

    def token_detection(self, token_map, position):
        # Функция определения ключевого слова
        """
        :param token_map: list of tuples of begin and end of every token
        :param position:  list of type: [int,int]
        :return: list of key word tokens position
        """
        # из за того что в начале стоит CLS позиции начала и конца ключевого слова сдвигаются на 5
        begin_postion = position[0]  # + 5
        end_position = position[1]  # + 5

        position_of_key_tokens = []
        for token_tuple in range(1, len(token_map) - 1):
            # if token is one
            if token_map[token_tuple][0] == begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

            # if we have multipli count of tokens for one key word
            if token_map[token_tuple][0] >= begin_postion and token_map[token_tuple][1] != end_position:
                position_of_key_tokens.append(token_tuple)
            if token_map[token_tuple][0] != begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

        return position_of_key_tokens

    def get_vector(self, input_ids_samp, token_type_ids_samp, attention_mask_samp):
        # Функция получения вектора ключевого слова
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_samp, token_type_ids=token_type_ids_samp,
                                 attention_mask=attention_mask_samp)
            hidden_states = outputs[2]
        # from [# layers, # batches, # tokens, # features] to [# tokens, # layers, # features]
        token_dim = torch.stack(hidden_states, dim=0)
        token_dim = torch.squeeze(token_dim, dim=1)
        token_dim = token_dim.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_dim:
            cat_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(cat_vec)

        return token_vecs_cat

    def get_avarage_embedding(self, embeddings_list, positions_list):
        # Функция получения среднего вектора
        avg_tensor = torch.stack((embeddings_list[positions_list[0]],))
        for i in range(1, len(positions_list)):
            avg_tensor = torch.cat((avg_tensor, embeddings_list[positions_list[i]].unsqueeze(0)))

        average_embedding = torch.mean(avg_tensor, 0)
        return average_embedding

    def vector_recognition(self, tokens_embeddings_ex, tokens_key_word_position_ex):
        # Функция подготовки вектора в зависимости от количества токенов,которым представляется ключевое слово
        if len(tokens_key_word_position_ex) > 1:
            embeddings_data = torch.tensor(
                self.get_avarage_embedding(tokens_embeddings_ex, tokens_key_word_position_ex))
        else:
            # print(tokens_embeddings_ex)
            # print(tokens_key_word_position_ex)
            embeddings_data = torch.tensor(tokens_embeddings_ex[tokens_key_word_position_ex[0]])
        return embeddings_data


class Trainer():
    def __init__(self, num_epochs=None, batch_size=None,
                 max_batches_per_epoch=None, early_stopping=10,
                 loss_fn=None, optimizer=None, model=None,
                 scheduler=None, device='cpu'):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_batches_per_epoch = max_batches_per_epoch
        self.early_stopping = early_stopping
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_model = model
        self.best_model = model

        self.train_loss = []
        self.valid_loss = []

    def predict(self, input_ids, input_mask, segment_ids):
        return self.best_model(input_ids, input_mask, segment_ids)

    def save_model(self, path: str):
        try:
            torch.save(self.best_model, path)
        except Exception as e:
            print(f"Не удалось сохранить модель. Ошибка {e}")
            exit(1)

        return True

    def load_model(self, path: str):
        try:
            self.best_model.load_state_dict(torch.load(path))
        except Exception as e:
            print(f"Не удалось загрузить модель. Ошибка {e}")
            exit(1)

        return True

    def fit(self, train_dataset, valid_dataset):

        device = torch.device(self.device)
        print(device)
        NerualNet = self.start_model
        NerualNet.to(device)

        NerualNet.train()

        self.optimizer = optim.Adam(NerualNet.parameters(), lr=0.0001)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size,
                                  shuffle=False, drop_last=True)

        best_val_loss = float('inf')  # Лучшее значение функции потерь на валидационной выборке

        best_ep = 0  # Эпоха, на которой достигалось лучшее значение функции потерь на валидационной выборке

        for epoch in range(self.num_epochs):
            start = dt.datetime.now()
            mean_loss = 0
            batch_n = 0
            for batch in train_loader:
                y_truth = batch["label"].float().to(device)
                input_ids = batch["input_ids"].to(device)
                input_mask = batch["input_mask"].to(device)
                segment_ids = batch["segment_ids"].to(device)

                y_pred = NerualNet(input_ids, input_mask, segment_ids).float()
                loss = self.loss_fn(y_pred, y_truth)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del batch
                torch.cuda.empty_cache()
                gc.collect()

                mean_loss += float(loss)
                batch_n += 1

            mean_loss /= batch_n
            self.train_loss.append(mean_loss)
            print(f'Эпоха: {epoch + 1}\n Train loss: {mean_loss}\n {dt.datetime.now() - start} сек.\n')

            NerualNet.eval()
            mean_loss = 0
            batch_n = 0
            with torch.no_grad():
                for batch in valid_loader:
                    if self.max_batches_per_epoch is not None:
                        if batch_n >= self.max_batches_per_epoch:
                            break

                input_ids = batch["input_ids"].to(device)
                input_mask = batch["input_mask"].to(device)
                segment_ids = batch["segment_ids"].to(device)
                target = batch["label"].float().to(device)

                predicted_values = NerualNet(input_ids, input_mask, segment_ids).float()
                loss = self.loss_fn(predicted_values, target)

                del batch
                torch.cuda.empty_cache()
                gc.collect()

                mean_loss += float(loss)
                batch_n += 1

            mean_loss /= batch_n
            self.valid_loss.append(mean_loss)
            print(f'Loss_val: {mean_loss}')

            if mean_loss < best_val_loss:
                self.best_model = NerualNet
                best_val_loss = mean_loss
                best_ep = epoch
            elif epoch - best_ep > self.early_stopping:
                print(f'{self.early_stopping} без улучшений. Прекращаем обучение...')
                break
            if self.scheduler is not None:
                scheduler.step()
            print()

        print("-=-=-=-=-=-=-=-=-=-= Evaluation of the best model =-=-=-=-=-=-=-=-=-=-")
        plt.plot(range(len(self.train_loss)), self.train_loss, color='green', label='train', linestyle='solid')
        plt.plot(range(len(self.valid_loss)), self.valid_loss, color='red', label='val', linestyle='solid')
        plt.legend()
        plt.show()

        with torch.no_grad():
            y_test = [float(sample['label']) for sample in valid_dataset]
            Y_pred = []
            Y_pred = [self.best_model(sample['input_ids'].unsqueeze(0).to(device),
                                      sample['input_mask'].unsqueeze(0).to(device),
                                      sample['segment_ids'].unsqueeze(0).to(device)) for sample in valid_dataset]
            Y_pred = [float(y > 0.5) for y in Y_pred]
            print()

            print(f"report: \n", classification_report(y_test, Y_pred))

def data_preparation(texts, definitions, labels, tokenizer, max_len):
    tokenizer = tokenizer
    feautures_X, feautures_Y = [], []

    for i, (text, definition) in enumerate(zip(texts, definitions)):
        text = tokenizer.tokenize(text)

        tokens = ["[CLS]"] + text + ["[SEP]"]
        text_segment_ids = [0] * (len(text) + 2)
        text_input_ids = tokenizer.convert_tokens_to_ids(tokens)
        text_input_mask = [1] * len(text_input_ids)

        padding_length = max_len - len(text_input_ids)
        text_input_ids += ([0] * padding_length)
        text_input_mask += ([0] * padding_length)
        text_segment_ids += ([0] * padding_length)

        definition = tokenizer.tokenize(definition)

        tokens = ["[CLS]"] + definition + ["[SEP]"]
        def_segment_ids = [0] * (len(definition) + 2)
        def_input_ids = tokenizer.convert_tokens_to_ids(tokens)
        def_input_mask = [1] * len(def_input_ids)

        padding_length = max_len - len(def_input_ids)
        def_input_ids += ([0] * padding_length)
        def_input_mask += ([0] * padding_length)
        def_segment_ids += ([0] * padding_length)

        feautures_X.append([text_input_ids, text_input_mask, text_segment_ids,
                            def_input_ids, def_input_mask, def_segment_ids])
        feautures_Y.append(labels[i])

    return feautures_X, feautures_Y



df = pd.read_csv('../../nn_data.csv')

max_len_text = df.text.str.len().max()
max_len_def = df.definition.str.len().max()

max_len = max_len_def
if max_len_text > max_len_def:
    max_len = max_len_text

data_X, data_Y = data_preparation(df.text,
                                  df.definition,
                                  df.label,
                                  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True),
                                  max_len)

train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size = 0.2, random_state=42)

train_dataset = DisambiguationDataset(train_X, train_Y)
test_dataset = DisambiguationDataset(test_X, test_Y)

# trainer = Trainer(num_epochs=40, batch_size=8, loss_fn=nn.BCELoss(), model=NerualNet(), device='cuda:0')
#
# trainer.fit(train_dataset=train_dataset,valid_dataset=valid_dataset)