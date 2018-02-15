from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, exposure, img_as_uint, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class d_set(Dataset):
	def __init__(self, root, csv_path, img_path, transform=None):
		"""
		Args:
			root (string): path to where images and annotations are
			csv_path (string): relative path to csv file (annotations)
			img_path (string): relative path to the folder where images are
			transform: pytorch transforms for transforms and tensor conversion
		"""
		# Read the csv file
		annotations_path = os.path.join(root,csv_path)
		self.data_info = pd.read_csv(annotations_path)
		self.img_path = os.path.join(root,img_path)  # Assign image path
		self.transform = transform  # Assign transform
		self.labels = np.asarray(self.data_info.iloc[:, 1:]) 
		
	def __getitem__(self, index):
		# Get label(class) of the image based on the cropped pandas column
		single_image_1hot = self.labels[index]
		# Get image name from the pandas df
		single_image_name = str(self.data_info.iloc[index][0]) + ".png"
		
		# Open image
		img = Image.open(os.path.join(self.img_path,single_image_name))

		# Transform image to tensor
		if self.transform is not None:
			img = self.transform(img)
		
		# Return image and the label
		return (img, single_image_1hot)

	def __len__(self):
		return len(self.data_info.index)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=5, padding=2), #28x28x16
			nn.ReLU(), #28x28x16
			nn.MaxPool2d(2)) #14x14x16
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, padding=2), #14x14x32
			nn.ReLU(),#14x14x32
			nn.MaxPool2d(2))#7x7x32
		self.fc = nn.Linear(7*7*32, 60) 
		self.fc2 = nn.Linear(60, 6)
		
	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out = self.fc2(out)
		return F.log_softmax(out, dim=1)

# Hyper Parameters
num_epochs = 1
batch_size = 100
learning_rate = 0.001

transformation = transforms.Compose([transforms.ToTensor()])
train_dataset = d_set(root="data/training",csv_path="annotations.csv",img_path="training_images",transform=transformation)
test_dataset = d_set(root="data/testing",csv_path="annotations.csv",img_path="testing_images",transform=transformation)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size, 
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size, 
										  shuffle=False)


model = Net()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
	# Train the Model
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images)
			#a label is a 1-hot array change it to a single class
			labels = Variable(torch.LongTensor(np.argmax(labels,axis=1)))

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			if (i+1) % 100 == 0:
				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
					   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
			writer.add_scalar('Train/Loss', loss.data[0], i)

def test():
	# Test the Model
	model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
	correct = 0
	total = 0
	it = 0
	for images, labels in test_loader:
		it += 1
		images = Variable(images)
		labels = Variable(torch.LongTensor(np.argmax(labels,axis=1)))
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels.data).sum()
		writer.add_scalar('Accuracy',(100 * correct / total), it)

	print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

train()
test()

#Save the Trained Model
torch.save(model.state_dict(), 'cnn.pkl')
