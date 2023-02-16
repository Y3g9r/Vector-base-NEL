import torch
import gc

class InputSample():
    def __init__(self, choices_features, label):
        input_ids, input_mask, segment_ids = choices_features
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def text_to_train_test_feautures(examples, labels, max_len):
    feautures_X = []
    feautures_Y = []
    feautures_train = []
    feautures_test = []

    for i, example in enumerate(examples):
        text = tokenizer.tokenize(example)

        tokens = ["[CLS]"] + text + ["[SEP]"]
        segment_ids = [0] * (len(text) + 2)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_len - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)

        feautures_X.append([input_ids, input_mask, segment_ids])
        feautures_Y.append(labels[i])

    skf = StratifiedKFold(n_splits=2)
    for train_index, test_index in skf.split(feautures_X, feautures_Y):
        X_train, X_test = np.array(feautures_X)[train_index.astype(int)], np.array(feautures_X)[test_index.astype(int)]
        y_train, y_test = np.array(feautures_Y)[train_index.astype(int)], np.array(feautures_Y)[test_index.astype(int)]

    for train_sample_X, train_sample_Y in zip(X_train, y_train):
        feautures_train.append(InputSample(
            choices_features=(train_sample_X[0], train_sample_X[1],
                              train_sample_X[2]),
            label=train_sample_Y
        ))

    for test_sample_X, test_sample_Y in zip(X_train, y_train):
        feautures_test.append(InputSample(
            choices_features=(test_sample_X[0], test_sample_X[1],
                              test_sample_X[2]),
            label=test_sample_Y
        ))

    return feautures_train, feautures_test


class DisasterDataset(Dataset):
    def __init__(self, data, pass_labels=True):
        self.samples = data
        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        items = {"input_ids": torch.tensor(self.samples[index].input_ids),
                 "input_mask": torch.tensor(self.samples[index].input_mask),
                 "segment_ids": torch.tensor(self.samples[index].segment_ids),
                 "label": torch.tensor(self.samples[index].label)}
        return items


class NerualNet(nn.Module):
    def __init__(self, hidden_size=768, num_class=2):
        super(NerualNet, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for layer in self.bert.encoder.layer[:11]:
            for param in layer.parameters():
                param.requires_grad = False
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(hidden_size, num_class)
        self.sigm = nn.Sigmoid()

    def forward(self, input_ids, input_mask, segment_ids):
        bert_output = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(bert_output[1].squeeze()))
            else:
                h += self.fc(dropout(bert_output[1].squeeze()))
        return h / len(self.dropouts)


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
                y_truth = batch["label"].long().to(device)
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
                target = batch["label"].long().to(device)

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



max_len = train_df.text.str.len().max()
train_feautures, valid_feautures = text_to_train_test_feautures(train_df['text'],
                                                    train_df['target'],max_len)

train_dataset = DisasterDataset(train_feautures)
valid_dataset = DisasterDataset(valid_feautures)



trainer = Trainer(num_epochs=40,batch_size=8,loss_fn=nn.CrossEntropyLoss()
,model=NerualNet(),device='cuda:0')

trainer.fit(train_dataset=train_dataset,valid_dataset=valid_dataset)