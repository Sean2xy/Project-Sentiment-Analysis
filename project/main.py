import pdb

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import load_data
from model import BertBaseUncased
from sklearn.metrics import f1_score,recall_score,precision_score
import matplotlib.pyplot as plt

# train_data = load_data("./dataset/augmented-processed-train.csv")
# test_data = load_data("./dataset/augmented-processed-test.csv")

train_data = load_data("./dataset/processed-train.csv")
test_data = load_data("./dataset/processed-test.csv")

tokenized_train, tokenized_test = [], []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text):
        self.text = text
        self.tokenized_train = []
        for each in self.text:
            text, label = each[0], each[1]
            text_encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt' # Return pytorch tensors.
            )
            self.tokenized_train.append([text_encoded_text, label])

    def __getitem__(self, index):
        ids = self.tokenized_train[index][0]['input_ids'][0]
        mask = self.tokenized_train[index][0]['attention_mask'][0]
        label = int(self.tokenized_train[index][1])

        return ids, mask, label

    def __len__(self):
        return len(self.text)


train_dataset = Dataset(train_data)
test_dataset = Dataset(test_data)

print("Finish Loading Data")

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda")
model = BertBaseUncased()
print("Finish Loading Model")
model.to(device)

epochs = 8

loss_function = nn.CrossEntropyLoss()

parameter_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {'params': [p for n, p in parameter_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in parameter_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

num_train_optimization_steps = len(train_loader) * epochs

optimizer = AdamW(optimizer_parameters, lr=2e-5)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_optimization_steps
)

# Metrics
train_loss = []
average_train_accuracy = []
test_accuracy = []
test_loss = []

test_F1 = []
test_Precision = []
test_Recall = []

save_path = './bestModelbase.pth'
best_accuracy = 0.0
train_steps = len(train_loader)
test_steps = len(test_loader)

for epoch in range(epochs):

    # Training
    model.train()
    training_sum_loss = 0.0
    train_accuracy = 0.0
    train_bitch = tqdm(train_loader)
    for step, data in enumerate(train_bitch):
        text, text_mask, labels = data
        optimizer.zero_grad()
        outputs = model(text.to(device), text_mask.to(device))

        #  Get the "logits" output by the model. The "logits" are the output
        #  values prior to applying an activation function like the softmax.

        loss = loss_function(outputs.logits, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        training_sum_loss += loss.item()
        train_predict_y = torch.max(outputs.logits, dim=1)[1]
        train_accuracy += torch.eq(train_predict_y, labels.to(device)).sum().item()
        train_bitch.desc = "epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                             epochs,
                                                             loss)
    # Evaluation
    model.eval()
    accuracy = 0.0
    testing_sum_loss = 0.0

    with torch.no_grad():
        testing_bitch = tqdm(test_loader)
        label_true = []
        label_predicted = []
        for test_data in testing_bitch:
            testing_text, testing_text_mask, testing_labels = test_data
            outputs = model(testing_text.to(device), testing_text_mask.to(device))

            testing_loss = loss_function(outputs.logits, testing_labels.to(device))
            testing_sum_loss += testing_loss.item()

            predict_y = torch.max(outputs.logits, dim=1)[1]
            accuracy += torch.eq(predict_y, testing_labels.to(device)).sum().item()
            label_true.extend(testing_labels.cpu().numpy().tolist())
            label_predicted.extend(predict_y.cpu().numpy().tolist())

    # Testing Results
    train_accuracy_rate = train_accuracy / (len(train_loader) * batch_size)
    test_accuracy_rate = accuracy / (len(test_loader) * batch_size)
    print('[epoch %d] train_loss: %.3f  test_loss: %.3f test_accuracy: %.3f train_accuracy: %.3f' %
          (epoch + 1, training_sum_loss / train_steps, testing_sum_loss / test_steps, test_accuracy_rate,
           train_accuracy_rate))

    f1 = f1_score(label_true, label_predicted)
    recall = recall_score(label_true,label_predicted)
    precision = precision_score(label_true,label_predicted)

    train_loss.append(training_sum_loss / train_steps)
    test_loss.append(testing_sum_loss / test_steps)
    average_train_accuracy.append(train_accuracy_rate)
    test_accuracy.append(test_accuracy_rate)

    test_F1.append(f1)
    test_Precision.append(precision)
    test_Recall.append(recall)

    if test_accuracy_rate >= best_accuracy:
        best_accuracy = test_accuracy_rate
        torch.save(model.state_dict(), save_path)

torch.save([train_loss, test_loss, average_train_accuracy, test_accuracy, test_F1,test_Recall,test_Precision], "metrics.pt")

metrics = torch.load("metrics.pt")

train_loss, test_loss, average_train_accuracy, test_accuracy, test_F1, test_Recall, test_Precision = metrics[0], metrics[1], metrics[2], metrics[3], \
                                                                        metrics[4], metrics[5], metrics[6]

iters = range(epochs)

plt.plot(iters, train_loss, 'b', label='train_loss')
plt.plot(iters, test_loss, 'g', label='test_loss')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc="upper left")
plt.show()

plt.plot(iters, test_accuracy, 'g', label='test_acc')
plt.plot(iters, average_train_accuracy, 'b', label='train_acc')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('accuracy rate')
plt.legend(loc="lower right")
plt.show()

plt.plot(iters, test_F1, 'b', label='test_F1')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('F1 score')
plt.legend(loc="lower right")
plt.show()

plt.plot(iters, test_Recall, 'b', label='test_Recall')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('Recall')
plt.legend(loc="lower right")
plt.show()

plt.plot(iters, test_Precision, 'b', label='test_Precision')
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.show()

print('Finish Training')