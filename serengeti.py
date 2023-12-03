import time
import torch
import torchvision
import itertools
import torchvision.transforms as T
from skimage import io
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DataParallel
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


log_file = 'log.txt'


batch_sizes = [8, 16, 32, 64, 128]
num_epochs_values = [25, 50, 75]
learning_rates = [0.01, 0.001, 0.0001]
num_workers_values = [6]

transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(degrees=(0,180)),
    T.ColorJitter(brightness=(0.5,1.0), contrast=1,saturation=0, hue=0.4),
    T.RandomCrop(size=(224,312)),
    T.Resize(224),
    T.ToTensor()
])


n = 0
train_l_sum = 0.0
train_acc_sum = 0.0
count = 0
train_losses = []
test_losses = []
train_correct = []
test_correct = []

start = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

param_combinations = list(itertools.product(batch_sizes, num_workers_values, num_epochs_values, learning_rates))


def log_to_file(log_file, message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
    log_message = f'{timestamp}{message}\n'
    with open(log_file, 'a') as f:
        f.write(log_message)

    print(log_message)

def evaluate_accuracy(data_iter, net, loss):
    acc_sum, n, l = torch.Tensor([0]), 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l += loss(y_hat, y).sum()
            acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.size()[0]
    return acc_sum.item() / n, l.item() / len(data_iter)


def train_validate(model, train_iter, test_iter, optimizer, loss, num_epochs):
    print('training on', device)
    for epoch in range(num_epochs):
        model.train()
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            optimizer.zero_grad()
            l = loss(y_pred, y).sum()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_pred.argmax(axis=1) == y).sum().item()
            n += y.size()[0]
        test_acc, test_loss = evaluate_accuracy(test_iter, model, loss)
        train_losses.append(train_l_sum / len(train_iter))
        train_correct.append(train_acc_sum / n)
        test_losses.append(test_loss)
        test_correct.append(test_acc)
        
        log_to_file(log_file, f'epoch {epoch + 1}, train loss {train_l_sum / len(train_iter):.4f}, train acc {train_acc_sum / n:.3f}, test loss {test_loss:.4f}, test acc {test_acc:.3f}, time {(time.time() - start) / 60:.1f} min')



def check_accuracy_test(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())         
        accuracy = float(num_correct) / float(num_samples) * 100
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro') 
        log_to_file(log_file, f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
        cm = confusion_matrix(y_true, y_pred)
        classes = np.arange(23)
        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        
        return

class AnimalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data.reset_index(inplace=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_count = self.data['label'].value_counts().to_dict()
        self.weight = [1.0 / (1.0 + self.label_count[label]) for label in self.data['label']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data['image_path_rel'].iloc[index]
        image = io.imread(img_name)
        y_label = torch.tensor(self.data['label'][index]) 
        if self.transform:
            image = self.transform(image)
            return image, y_label

        return image, y_label
    

DataParallel.device_ids = [0,1]
model = DataParallel(torchvision.models.resnet18(pretrained=True))
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args['learning_rate'])


treino = pd.read_csv('training_data_d1.csv')
val = pd.read_csv('validation_data_d1.csv')


train_dataset = AnimalImageDataset(csv_file = 'training_data_d1.csv', root_dir = '', transform = transform)
val_dataset = AnimalImageDataset(csv_file = 'validation_data_d1.csv', root_dir = '', transform = transform)

sampler_train = WeightedRandomSampler(weights=train_dataset.weight, num_samples=len(train_dataset), replacement=True)

def load_data():
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = args['batch_size'],
                              sampler=sampler_train,
                              pin_memory = True,
                              num_workers=6)

    val_loader =  DataLoader(dataset=val_dataset,
                             batch_size = args['batch_size'],
                             pin_memory = True,
                             num_workers=6)

    return train_loader, val_loader


##########################################################################################################

for i, params in enumerate(param_combinations, start=1):
    args = {
        'batch_size': params[0],
        'num_workers': params[1],
        'num_epochs': params[2],
        'learning_rate': params[3]
    }

    log_to_file(log_file, f'Treinamento {i}\n')
    log_to_file(log_file, f'Args {args}\n')

    train_iter, val_iter = load_data()
    start_time = time.time()
    train_validate(model, train_iter, val_iter, optimizer, criterion, args['num_epochs'])
    log_to_file(log_file, f'Tempo de treinamento: {((time.time() - start_time)/60):.2f} minutos')


    teste_data = AnimalImageDataset(csv_file = "testing_data_d1.csv",
                                    root_dir = '',
                                    transform = transform)

    teste_loader = DataLoader(dataset=teste_data,
                            batch_size = args['batch_size'],
                            shuffle=False)


    check_accuracy_test(teste_loader, model)

    figure(figsize=(6, 4), dpi=100)
    plt.plot(train_losses, label = 'Training loss')
    plt.plot(test_losses, label = 'Test loss')
    plt.title('Loss at end of each epoch')
    plt.legend()
    plt.savefig('saida_' + count + '.png')

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

##########################################################################################################



