#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[8]:


# Loading the datasets and data augmentations
# Data Augmentation step

torch.manual_seed(2020802018)
train_transform = transforms.Compose(
    [transforms.RandomVerticalFlip(0.4), # Randomly Vertical Flip
     transforms.RandomHorizontalFlip(0.4), # Randomly Horizontal Flip
     transforms.RandomCrop(32, padding=2), # Random Crop
     transforms.ToTensor(), # Converting PILI image to tensors
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #To transform PILImage ([0,1]) images from torchvision into tensors, and normalize into [-1,1]
test_transform = transforms.Compose(
     [transforms.ToTensor(), # Converting PILI image to tensors
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #To transform PILImage ([0,1]) images from torchvision into tensors, and normalize into [-1,1]

batch_size = 64 # Defining the mini-batch size

# Defining train and test data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = train_transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2) # Shuffling the data and apply data augmentation techniques
or_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                         shuffle=False, num_workers=2) # Loader for accuracy calculation
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

fe_testloader = torch.utils.data.DataLoader(testset, batch_size=testset.__len__(), 
                                         shuffle=False, num_workers=2) # Loader for t-Sne plots
# Defining classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[9]:


# Visualising some sample images
torch.manual_seed(2020802018)

# Kernel die problem solution 
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Plot function
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Some random training images
plot_batch0 = 5 # Defining number of images to show
plotloader0 = torch.utils.data.DataLoader(testset, batch_size=plot_batch0, 
                                         shuffle=False, num_workers=2) # Loader for sample estimations
dataiter = iter(plotloader0)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(plot_batch0)))


# In[215]:


# Visualization of Sample Randomly cropped and flipped images
torch.manual_seed(2020802018)
sample_image, sample_label = iter(torch.utils.data.DataLoader(testset, batch_size=1, 
                                         shuffle=True, num_workers=2)).next()
imshow(torchvision.utils.make_grid(sample_image)) # Original Image
print("Original Image  =",classes[sample_label])

transform_vflip = transforms.Compose([transforms.RandomVerticalFlip(0.99)])
imshow(transform_vflip(np.squeeze(sample_image)))
print("Vartical Flipped Image")

transform_hflip = transforms.Compose([transforms.RandomHorizontalFlip(0.99)])
imshow(transform_hflip(np.squeeze(sample_image)))
print("Horizontal Flipped Image")

transform_crop = transforms.Compose([transforms.RandomCrop(28, padding=4)])
imshow(transform_crop(np.squeeze(sample_image)))
print("Cropped Image")


# In[175]:


# Defining Convolutional Neural Network
torch.manual_seed(2020802018)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.BN1 = nn.BatchNorm2d(num_features = 16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.BN2 = nn.BatchNorm2d(num_features = 32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.BN3 = nn.BatchNorm2d(num_features = 64)
        self.fc1 = nn.Linear(in_features = 64 * 6 * 6, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 32)
        self.fc4 = nn.Linear(in_features = 32, out_features = 10)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.BN1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.BN2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.BN3(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        y = x.clone().detach() # Feature vectors after the flatten
        x = F.relu(self.fc1(x)) # Fully-connected layer 1 (120-dimensional)
        x = F.relu(self.fc2(x)) # Fully-connected layer 2 (120-dimensional)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


net = Net()


# In[176]:


# Defining Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.011, momentum=0.8)
#optimizer = optim.Adam(net.parameters(), lr=0.001)


# In[177]:


# Train the Network

Loss = np.empty((0,1))
num_epochs = 60

inputs_t, labels_t= next(iter(fe_testloader)) # Defining inputs and labels for t-SNE plot
for epoch in range(num_epochs):  # loop over the dataset multiple times
    Mini_Loss = np.empty((0,1)) # Storing loss values for each mini-batch (Refreshed for each epoch)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        Mini_Loss = np.append(Mini_Loss,loss.item())
        # Print Epoch, mini-batch and corresponding Loss value
        print( "Epoch =", epoch + 1, "/",num_epochs,  ",", "Mini-batch =", i + 1,",", "Current Loss =", loss.item())      
    Loss = np.append(Loss,Mini_Loss.mean())
    if epoch == 0: # Extracting feature vectors (for the test data)
        y1 = net(inputs_t)[1] # In the beginning of the training
    if epoch == num_epochs/2:
        y2 = net(inputs_t)[1] # In the middle of the training
    if epoch == num_epochs - 1: 
        y3 = net(inputs_t)[1] # In the end of the training
print('Training is completed')


# In[178]:


# Plotting Loss Function
plt.title("Loss Function") 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.plot(Loss)
plt.savefig('Loss_plot.png')
plt.show()


# In[179]:


# Saving the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# In[180]:


## Sample estimations on the test set

torch.manual_seed(2020802018)
# Plot function and test for sample 5 images
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

plot_batch = 5 # Defining number of images to show
plotloader = torch.utils.data.DataLoader(testset, batch_size=plot_batch, 
                                         shuffle=True, num_workers=2) # Loader for sample estimations
dataiter = iter(plotloader)
images, labels = dataiter.next()

# Plotting sample images
imshow(torchvision.utils.make_grid(images))
print('Ground Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(plot_batch)))

# Loading the saved model
net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)[0] #The outputs for 5 test images 
_, predicted = torch.max(outputs, 1) # Selecting the class with the highest probability

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(plot_batch)))


# In[181]:


# Measuring the model performance on the train data
torch.manual_seed(2020802018)
def train_acc(trainloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in or_trainloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)[0]
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 50000 train images: %d %%' % (
        100 * correct / total))
    
train_acc(trainloader, net)


# In[183]:


# Measuring the model performance on the test data
torch.manual_seed(2020802018)

def test_acc(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)[0]
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
test_acc(testloader, net)


# In[184]:


# Measuring the performance for each class
torch.manual_seed(2020802018)

def class_acc(classes, testloader, net):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)[0]
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                       accuracy))
        
class_acc(classes, testloader, net)


# In[185]:


# T-SNE Datasets (In the and end of the training)
from sklearn.manifold import TSNE

m = TSNE(random_state = 1234) # TSNE model (This operation takes approximately 9 minutes)
tsne_features = m.fit_transform(y3.numpy())


# In[186]:


# Plotting the latent space of the model - In the END of training

plt.figure(figsize=(155,12))
tx = tsne_features[:,0]
ty = tsne_features[:,1]

X = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
Y = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

cdict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'brown', 7: 'gray', 8: 'pink', 9: 'purple'}
fig, ax = plt.subplots(figsize=(18,12))
for g in np.unique(labels_t.numpy()):
    ix = np.where(labels_t.numpy() == g)
    ax.scatter(X[ix], Y[ix], c = cdict[g], label = classes[g], s = 20)
ax.legend()
plt.title("Latent Space of the Model for the Test Data - END")
plt.savefig("tsne_end.png")
plt.show()


# In[187]:


# T-SNE Datasets (In the beginning of the training)
m = TSNE(random_state = 1234) # TSNE model (This operation takes approximately 3 minutes)
tsne_features_1 = m.fit_transform(y1.numpy()) # beginning


# In[188]:


# Plotting the latent space of the model - In the BEGINNING of training

plt.figure(figsize=(155,12))
tx = tsne_features_1[:,0]
ty = tsne_features_1[:,1]

X = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
Y = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

cdict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'brown', 7: 'gray', 8: 'pink', 9: 'purple'}
fig, ax = plt.subplots(figsize=(18,12))
for g in np.unique(labels_t.numpy()):
    ix = np.where(labels_t.numpy() == g)
    ax.scatter(X[ix], Y[ix], c = cdict[g], label = classes[g], s = 20)
ax.legend()
plt.title("Latent Space of the Model for the Test Data - BEGINNING")
plt.savefig("tsne_begin.png")
plt.show()


# In[189]:


# T-SNE Datasets (In the middle of the training)
m = TSNE(random_state = 1234) # TSNE model (This operation takes approximately 3 minutes)
tsne_features_2 = m.fit_transform(y2.numpy()) # middle


# In[190]:


# Plotting the latent space of the model - In the MIDDLE of training

plt.figure(figsize=(155,12))
tx = tsne_features_2[:,0]
ty = tsne_features_2[:,1]

X = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
Y = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

cdict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange', 6: 'brown', 7: 'gray', 8: 'pink', 9: 'purple'}
fig, ax = plt.subplots(figsize=(18,12))
for g in np.unique(labels_t.numpy()):
    ix = np.where(labels_t.numpy() == g)
    ax.scatter(X[ix], Y[ix], c = cdict[g], label = classes[g], s = 20)
ax.legend()
plt.title("Latent Space of the Model for the Test Data - MIDDLE")
plt.savefig("tsne_middle.png")
plt.show()


# In[123]:


# Trying another optimizer (Adam)

# Again running this block
torch.manual_seed(2020802018)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.BN1 = nn.BatchNorm2d(num_features = 16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.BN2 = nn.BatchNorm2d(num_features = 32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.BN3 = nn.BatchNorm2d(num_features = 64)
        self.fc1 = nn.Linear(in_features = 64 * 6 * 6, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 32)
        self.fc4 = nn.Linear(in_features = 32, out_features = 10)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.BN1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.BN2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.BN3(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        y = x.clone().detach() # Feature vectors after the flatten
        x = F.relu(self.fc1(x)) # Fully-connected layer 1 (120-dimensional)
        x = F.relu(self.fc2(x)) # Fully-connected layer 2 (120-dimensional)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


net = Net()


# In[ ]:


# Trying Another optimizer (Before running this code block, the network should be initialized)
# Defining Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.011)

# Train the Network

Loss = np.empty((0,1))
num_epochs = 60

inputs_t, labels_t= next(iter(fe_testloader)) # Defining inputs and labels for t-SNE plot
for epoch in range(num_epochs):  # loop over the dataset multiple times
    Mini_Loss = np.empty((0,1)) # Storing loss values for each mini-batch (Refreshed for each epoch)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        Mini_Loss = np.append(Mini_Loss,loss.item())
        # Print Epoch, mini-batch and corresponding Loss value
        print( "Epoch =", epoch + 1, "/",num_epochs,  ",", "Mini-batch =", i + 1,",", "Current Loss =", loss.item())      
    Loss = np.append(Loss,Mini_Loss.mean())
    if epoch == 0: # Extracting feature vectors (for the test data)
        y1 = net(inputs_t)[1] # In the beginning of the training
    if epoch == num_epochs/2:
        y2 = net(inputs_t)[1] # In the middle of the training
    if epoch == num_epochs - 1: 
        y3 = net(inputs_t)[1] # In the end of the training
print('Training is completed')


# In[ ]:


# Measuring the model performance on the train and test data with different optimizer (Adam)
torch.manual_seed(2020802018)
   
train_acc(trainloader, net)
test_acc(testloader, net)

