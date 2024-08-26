import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import torch.optim as optim
import streamlit as st
import zipfile
import torchvision
from torch.utils.data import DataLoader, random_split
import os
from io import BytesIO
from PIL import Image
import shutil

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.406, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.406, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def load_mode(class_names, device):
  model = models.resnet18(pretrained=True)
  model_feat = model.fc.in_features
  model.fc = nn.Linear(model_feat, len(class_names))
  model = model.to(device)
  return model

def save_model(model, path):
  torch.save(model.state_dict(), path)

def load_save_model(model, path, device):
  model.load_state_dict(torch.load(path))
  model = model.to(device)
  return model

def load_dataset(data_dir, transform):
  dataset = datasets.ImageFolder(data_dir, transform)
  train_len = int(0.8 * len(dataset))
  val_len = len(dataset) - train_len
  train_set, val_set = random_split(dataset, [train_len, val_len])
  return train_set, val_set

def test_own_image(image_path, model, transform, class_names, device):
  pred = {}
  image = Image.open(image_path)
  image = transform(image).unsqueeze(0).to(device)
  model.eval()
  with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)
  return class_names[pred.item()]

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        return True
    else:
        return False


pages = ["Training", "Test Own Image"]
choice = st.selectbox("Select a page", pages)

data_dir = 'dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 12

if choice == "Training":
  st.header("Training")
  upload_file = st.file_uploader("upload zip file dataset", type=['zip'], accept_multiple_files=True)

  if upload_file:
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    for upload in upload_file:
      with zipfile.ZipFile(BytesIO(upload.read()), 'r') as f:
        f.extractall(data_dir)

    train_dataset, val_dataset = load_dataset(data_dir, transform=transform['train'])
    data_loaders = {'train_loader' : DataLoader(train_dataset, batch_size=5, shuffle=True),
                    'test_loader': DataLoader(val_dataset, batch_size=5, shuffle=False)}

    class_names = os.listdir(data_dir)
    model = load_mode(class_names, device)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    st.write(f"device: {device}")
    if st.button("start training"):
      st.write("start training...")
      correct = 0
      total = 0

      for epoch in range(num_epochs):
        for images, labels in data_loaders['train_loader']:
          images = images.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          outputs = model(images)
          _, pred = torch.max(outputs, 1)
          correct += (pred == labels).sum().item()
          total += labels.size(0)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          accur = 100 * correct / total

        st.write(f"epoch [{epoch+1} / {num_epochs}], loss: {loss.item():.4f}, accurracy: {accur:.4f}")


      st.write("start testing...")
      with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loaders['test_loader']:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, pred = torch.max(outputs, 1)
          correct += (pred == labels).sum().item()
          total += labels.size(0)
          accur = 100 * correct / total
          st.write(f"test accurracy{accur:.4f}")

      save_model(model, 'model.pth')
      st.write("model saved successfully...")

    
if choice == "Test Own Image":
  st.header("Test Own Image")

  image_path = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

  if image_path:
    st.write("uploaded images")
    pred = {}
    class_names = sorted([d for d in os.listdir('dataset') if os.path.isdir(os.path.join(data_dir, d))])
    model = load_mode(class_names, device)
    model = load_save_model(model, 'model.pth', device)

    for img in image_path:
      st.image(img, caption='Uploaded Image.', use_column_width=True)
      prediction = test_own_image(img, model, transform['val'], class_names, device)
      pred[img.name] = prediction

    if st.button("Classify Image"):
      st.write("Classifying...")
      for img_name, pred in pred.items():
        st.write("## Image: {}".format(img_name))
        st.write("## Prediction: {}".format(pred))

st.write("""
# Note: \n
    1.create a folder and that folder name should be what are images(class name) stored(ant, bees, trees) at least 100 images\n
    2.image folder contains only same type of images ex: if you store folder ants images then store only ants images that folder \n
    3.convert folder into zip file and upload it(.zip only). you can upload multiple file also (not recommend because run this project with "CPU") \n
    4.select page option train images and test images \n
    5.Must train with images(classes) and test images (you can select multiple images for testing) \n
    6.This model sometime missclassify or wrong
"""
)
passwd = st.sidebar.text_input("password", type='password')
if passwd == 'hacker4321':
    
    st.sidebar.write(os.listdir(data_dir))
    if st.sidebar.button("Delete Dataset Folder"):
        folder_path = st.sidebar.text_input("folder name")
        try:
            if folder_path in  os.listdir(data_dir) and delete_folder(data_dir + '/' + folder_path):
                st.sidebar.write(f"Folder '{data_dir}' has been deleted successfully.")
            else:
                st.sidebar.error(f"Folder '{data_dir}' does not exist or could not be deleted.")
        except FileNotFoundError:
            st.sidebar.write("folder not found")
