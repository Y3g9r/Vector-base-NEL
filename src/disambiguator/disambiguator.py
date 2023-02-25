import torch
import torch.nn as nn
from torch.utils.data import  Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import ast
import datetime as dt
import gc

class DisambiguationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        items = {"text_input_ids": torch.tensor(self.samples[index][0]),
                 "text_input_mask": torch.tensor(self.samples[index][1]),
                 "text_segment_ids": torch.tensor(self.samples[index][2]),
                 "text_offset_mapping": torch.tensor(self.samples[index][3]),
                 "text_pos": torch.tensor(self.samples[index][4]),
                 "def_input_ids": torch.tensor(self.samples[index][5]),
                 "def_input_mask": torch.tensor(self.samples[index][6]),
                 "def_segment_ids": torch.tensor(self.samples[index][7])}
        return items

def data_preparation(texts, definitions, position, tokenizer, max_len):
    tokenizer = tokenizer
    feautures_X, feautures_Y = [], []

    for i, (text, definition) in enumerate(zip(texts, definitions)):
        text = tokenizer(text, return_offsets_mapping=True,max_length=max_len,truncation=True,padding='max_length')

        text_input_ids = text["input_ids"]
        text_input_mask = text["attention_mask"]
        text_segment_ids = text["token_type_ids"]
        text_offset_mapping = text["offset_mapping"]
        text_pos = [position[i]]

        definition = tokenizer(definition, return_offsets_mapping=True,max_length=max_len,padding='max_length',truncation=True)

        def_input_ids = definition["input_ids"]
        def_input_mask = definition["attention_mask"]
        def_segment_ids = definition["token_type_ids"]

        feautures_X.append([text_input_ids, text_input_mask, text_segment_ids, text_offset_mapping,
                            text_pos, def_input_ids, def_input_mask, def_segment_ids])

    return feautures_X


class NerualNet(nn.Module):
    def __init__(self, hidden_size=768, max_seq_len=388, device='cpu'):
        self.device = device
        super(NerualNet, self).__init__()
        self.bert = BertModel.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru', output_hidden_states=True,
                                              return_dict=False)

        for layer in self.bert.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False

        self.text_linear_1 = torch.nn.Linear(1024, 512)
        self.def_linear_1 = torch.nn.Linear(1024, 512)

        self.Dropout_text_1 = torch.nn.Dropout(0.5)
        self.Dropout_def_1 = torch.nn.Dropout(0.5)

        self.text_linear_2 = torch.nn.Linear(512, 256)
        self.def_linear_2 = torch.nn.Linear(512, 256)

        self.Dropout_text_2 = torch.nn.Dropout(0.5)
        self.Dropout_def_2 = torch.nn.Dropout(0.5)

        self.text_linear_3 = torch.nn.Linear(256, 128)
        self.def_linear_3 = torch.nn.Linear(256, 128)

        self.Dropout_text_3 = torch.nn.Dropout(0.5)
        self.Dropout_def_3 = torch.nn.Dropout(0.5)

        self.text_linear_4 = torch.nn.Linear(128, 128)
        self.def_linear_4 = torch.nn.Linear(128, 128)

        self.Dropout_text_4 = torch.nn.Dropout(0.5)
        self.Dropout_def_4 = torch.nn.Dropout(0.5)

        self.sigm_linear_1 = torch.nn.Linear(128, 1)
        self.Dropout_sigm = torch.nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_input_ids, text_input_mask, text_segment_ids, text_offset_mapping,
                text_pos, def_input_ids, def_input_mask, def_segment_ids):

        embd_batch = torch.tensor([[[], []]]).to(self.device)
        first_pass = False
        for i in range(len(text_input_ids)):
            # получаем эмбединги ключевого слова из примера употребления
            examples_token_key_word_position = self.token_detection(text_offset_mapping[i], text_pos[i][0])
            example_token_vec = self.get_vector(text_input_ids[i], text_segment_ids[i], text_input_mask[i])
            example_embeddings = self.vector_recognition(example_token_vec, examples_token_key_word_position)

            # получаем эмбединг определения
            def_embedding = self.get_defenition_embedding(def_input_ids[i], def_segment_ids[i],
                                                          def_input_mask[i]).squeeze(0)
            # объединяем два вектора в 1 и добавляем в общий массив (получаем тензор 2x768)
            embd_sample = torch.stack((example_embeddings, def_embedding)).to(self.device)
            if not first_pass:
                embd_batch = torch.cat((embd_batch, embd_sample.unsqueeze(0)), -1)
                first_pass = True
            else:
                embd_batch = torch.cat((embd_batch, embd_sample.unsqueeze(0)), 0)

        text_emb = embd_batch[:, 0, :]
        def_emb = embd_batch[:, 1, :]

        ex_emb = self.Dropout_text_4(self.text_linear_4(self.Dropout_text_3(self.text_linear_3(
            self.Dropout_text_2(self.text_linear_2(self.Dropout_text_1(self.text_linear_1(text_emb))))))))
        def_emb = self.Dropout_def_4(self.def_linear_4(self.Dropout_def_3(
            self.def_linear_3(self.Dropout_def_2(self.def_linear_2(self.Dropout_def_1(self.def_linear_1(def_emb))))))))

        dist = torch.abs(ex_emb - def_emb)
        py = self.Dropout_sigm(self.sigm_linear_1(dist))

        y = self.sigmoid(py).permute(1, 0).squeeze(0)

        return y

    def get_defenition_embedding(self, def_input_ids, def_segment_ids, def_input_mask):
        """
        Функция получения вектора дефенишина сущности
        :param def_input_ids:
        :param def_segment_ids:
        :param def_input_mask:
        :return: bert pooler output vector
        """
        with torch.no_grad():
            output = self.bert(input_ids=def_input_ids.unsqueeze(0), token_type_ids=def_segment_ids.unsqueeze(0),
                               attention_mask=def_input_mask.unsqueeze(0))
        hidden_states = output[1]
        return hidden_states

    def token_detection(self, token_map, position):
        """
        Функция определения ключевого слова
        :param token_map: list of tuples of begin and end of every token
        :param position:  list of type: [int,int]
        :return: list of key word tokens position
        """
        # из за того что в начале стоит CLS позиции начала и конца ключевого слова сдвигаются на 5
        begin_postion = position[0]  # + 5
        end_position = position[1]  # + 5

        position_of_key_tokens = []
        for token_tuple in range(1, len(token_map) - 1):
            # Если ключевое слово представляется одним токеном
            if token_map[token_tuple][0] == begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

            # Если ключевое слово представляется несколькими токенами
            if token_map[token_tuple][0] >= begin_postion and token_map[token_tuple][1] != end_position:
                position_of_key_tokens.append(token_tuple)
            if token_map[token_tuple][0] != begin_postion and token_map[token_tuple][1] == end_position:
                position_of_key_tokens.append(token_tuple)
                break

        return position_of_key_tokens

    def get_vector(self, input_ids_samp, token_type_ids_samp, attention_mask_samp):
        """
        Функция получения вектора ключевого слова
        :param input_ids_samp:
        :param token_type_ids_samp:
        :param attention_mask_samp:
        :return:
        """
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids_samp.unsqueeze(0), token_type_ids=token_type_ids_samp.unsqueeze(0),
                                attention_mask=attention_mask_samp.unsqueeze(0))
        hidden_states = outputs[2]

        # из [# layers, # batches, # tokens, # features]
        # в [# tokens, # layers, # features]
        token_dim = torch.stack(hidden_states, dim=0)
        token_dim = torch.squeeze(token_dim, dim=1)
        token_dim = token_dim.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_dim:
            cat_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(cat_vec)
        return token_vecs_cat

    def vector_recognition(self, tokens_embeddings_ex, tokens_key_word_position_ex):
        """
        Функция подготовки вектора в зависимости от количества токенов,которым представляется ключевое слово
        :param tokens_embeddings_ex:
        :param tokens_key_word_position_ex:
        :return:
        """
        if len(tokens_key_word_position_ex) > 1:
            embeddings_data = torch.tensor(
                self.__get_avarage_embedding(tokens_embeddings_ex, tokens_key_word_position_ex))
        else:
            embeddings_data = torch.tensor(tokens_embeddings_ex[tokens_key_word_position_ex[0]])
        return embeddings_data

    def __get_avarage_embedding(self, embeddings_list, positions_list):
        """
        Функция получения среднего вектора (применяется в случае если ключевое слово состоит из нескольких токенов)
        :param embeddings_list:
        :param positions_list:
        :return:
        """
        avg_tensor = torch.stack((embeddings_list[positions_list[0]],))
        for i in range(1, len(positions_list)):
            avg_tensor = torch.cat((avg_tensor, embeddings_list[positions_list[i]].unsqueeze(0)))

        average_embedding = torch.mean(avg_tensor, 0)
        return average_embedding


def make_predict(texts: list, positions: list, definitions: list, device="cpu"):
    """
    Делает предикты
    :param texts:
    :param positions:
    :param definitions:
    :param device:
    :return:
    """
    texts_list = texts
    postions_list = positions
    definitions_list = definitions

    pd_texts = []
    pd_positions = []
    pd_definitions = []
    for i in range(len(definitions_list)):
        for j in range(len(definitions_list[i])):
            for k in range(len(definitions_list[i][j])):
                pd_texts.append(str(texts_list[i]))
                pd_positions.append(postions_list[i][j])
                pd_definitions.append(str(definitions_list[i][j][k]))

    df = pd.DataFrame({'text': pd_texts, 'positions': pd_positions, 'definitions': pd_definitions})

    max_len_texts = df.text.str.len().max()
    max_len_defs = df.definitions.str.len().max()

    max_len = max_len_defs
    if max_len_texts > max_len_defs:
        max_len = max_len_texts

    with torch.no_grad():
        model = NerualNet(max_seq_len=max_len, device=device)
        model.eval()
        model.load_state_dict(torch.load("./../disambiguator/modelN72f.pth"))
        model.to(device)
        predicted_values=[]
        data_x = data_preparation(df.text,
                                  df.definitions,
                                  df.positions,
                                  BertTokenizerFast.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru',
                                                                    do_lower_case=True),
                                  max_len)
        dataset = DisambiguationDataset(data_x)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        for sample in loader:
            text_input_ids = sample["text_input_ids"].to(device)
            text_input_mask = sample["text_input_mask"].to(device)
            text_segment_ids = sample["text_segment_ids"].to(device)
            text_offset_mapping = sample["text_offset_mapping"].to(device)
            text_pos = sample["text_pos"].to(device)
            def_input_ids = sample["def_input_ids"].to(device)
            def_input_mask = sample["def_input_mask"].to(device)
            def_segment_ids = sample["def_segment_ids"].to(device)

            predicted = model(text_input_ids, text_input_mask, text_segment_ids, text_offset_mapping,
                                         text_pos, def_input_ids, def_input_mask, def_segment_ids).float()
            predicted_values.append(predicted.cpu().detach().numpy())
        df['predicted'] = predicted_values

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(df)

