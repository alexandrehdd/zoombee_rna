import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import torchvision.transforms as T
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


treino = pd.read_csv('training_data_d1.csv')
teste = pd.read_csv('testing_data_d1.csv')
val = pd.read_csv('validation_data_d1.csv')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = 'model_resnet50_pretrained.pth'


model_state_dict = torch.load(model_path, map_location=device)
model = torchvision.models.resnet50()
model.load_state_dict(model_state_dict)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

image_paths = dt['path']
new_features = []
image_tensor_array = []

for image_path in image_paths:
    image = Image.open(image_path)

    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image = Image.fromarray(image_np)
        image = transform(image)
    else:
        image = transform(image)

    image_tensor = image.unsqueeze(0).to(device)
    image_tensor_array.append(image_tensor)

    features = model(image_tensor)
    features_cpu = features.squeeze().detach().cpu().numpy()

    new_features.append(features_cpu)

new_features = StandardScaler().fit_transform(new_features)


transform = transforms.Compose([
    T.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = AnimalImageDataset(csv_file='new_metadata_images_with_crop_02_12_23.csv',
                             root_dir='',
                             transform=transform)
features = []
labels = []
for i in range(len(dataset)):
    image, label_numeric = dataset[i]
    features.append(image)
    labels.append(label_numeric)

X_train, X_test, y_train, y_test = train_test_split(new_features, labels, test_size=0.4, random_state=42)
print(X_train.shape, X_test.shape)

scaler = StandardScaler()

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
predictions = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X_test, y_test)
print(accuracy)

predictions = svm_classifier.predict(X_test)
report = classification_report(y_test, predictions)
print("Relatório de Classificação:\n", report)


cm = confusion_matrix(y_test, predictions)
cmap = plt.get_cmap('Blues')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(9, 9))
cm_display.plot(ax=ax, values_format='.4g', cmap=cmap)
plt.show()
plt.savefig('confusion_matrix.png')
