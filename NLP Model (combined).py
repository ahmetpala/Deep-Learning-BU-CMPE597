
# Network Class
import numpy as np


# One Hot Representation Function

def one_hot(indice):
    array = np.zeros(shape=(1, 250))
    for i in range(len(array)):
        array[0, indice] = 1
    return array


# Activations Functions
def sigmoid(x):  # Sigmoid Function
    return 1 / (1 + np.exp(-x))


def softmax(array):  # Softmax function
    n_array = np.copy(array)
    return (np.exp(n_array) / sum(np.exp(n_array)))


# Derivation Functions
def derive_ln(array):  # Derivation of natural logarithm
    n_array = np.copy(array)
    return 1 / array_n


def derive_softmax(array):  # Derivation of softmax function
    array_n = np.copy(array)
    W = np.zeros(shape=(len(array_n), len(array_n)))
    for i in range(len(array_n)):
        for j in range(len(array_n)):
            if i == j:
                W[i, j] = array_n[i] * (1 - array_n[j])
            else:
                W[i, j] = -array_n[j] * array_n[i]
    return W


def derive_sigmoid(array):  # Derivation of natural logarithm
    n_array = np.copy(array)
    return np.multiply(sigmoid(n_array), (1 - sigmoid(n_array)))


def relu(x):
    return np.maximum(0, x)


def derive_relu(x):
    return np.maximum(0, x) / np.abs(x)


# Mini-batch Algorithm
# Generating input function from original indexes to combined one hot representations
def gen_mbatch(minibatch_indexes):
    if minibatch_indexes.shape[1] > 1:
        final_matrix = np.zeros(shape=(250, 3 * len(minibatch_indexes)))
        for i in range(minibatch_indexes.shape[0]):
            for j in range(minibatch_indexes.shape[1]):
                final_matrix[:, (3 * i + j)] = one_hot(minibatch_indexes[i, j])
    else:
        final_matrix = np.zeros(shape=(250, len(minibatch_indexes)))
        for i in range(minibatch_indexes.shape[0]):
            final_matrix[:, i] = one_hot(minibatch_indexes[i, 0])

    return final_matrix


# Data shuffling and minibatch generation function

# def gen_minibatch(train_data, test_data, minibatch_size):
#    mini_batches = []
#    combined = np.column_stack((train_data, test_data))
#    np.random.shuffle(combined)
#    batch_number = train_data.shape[0]//minibatch_size
#    for i in range(batch_number+1):
#        if len(combined) > minibatch_size:
#            m_batch = combined[:minibatch_size,]
#            combined = np.delete(combined, range(minibatch_size), 0)
#            mini_batches.append((m_batch))
#        else:
#            m_batch = combined
#            mini_batches.append((m_batch))
#            break
#    return mini_batches

def gen_minibatch(train_data, test_data, minibatch_size):
    mini_batches = []
    combined = np.column_stack((train_data, test_data))
    np.random.shuffle(combined)
    batch_number = train_data.shape[0] // minibatch_size
    for i in range(batch_number + 1):
        if len(combined) > minibatch_size:
            m_batch = combined[:minibatch_size, ]
            combined = np.delete(combined, range(minibatch_size), 0)
            mini_batches.append((m_batch))
        else:
            m_batch = combined
            mini_batches.append((m_batch))
            done = True
            for j in range(len(mini_batches)):
                minn = np.argmax(mini_batches[j][:, [3]])
                maxx = np.argmin(mini_batches[j][:, [3]])
                if maxx == minn:
                    done = False
            if done:
                break
            else:
                mini_batches = gen_minibatch(train_data, test_data, minibatch_size)
    return mini_batches


# In[8]:


# Forward Propogation Function

def forward_propogation(x_ar, W1, W2, W3, b2, b3):
    # Forward Propogation
    # Layer 1 (Embedding Layer)
    # a1 = []
    # for i in range(x_ar.shape[1]//3):
    #    a1.append(np.concatenate((np.dot(W1,x_ar[:,3*i]),np.dot(W1,x_ar[:,(3*i+1)]),np.dot(W1,x_ar[:,(3*i+2)]))))
    # a1 = np.array(a1).T
    a1 = np.concatenate(((np.dot(W1, x_ar[:, 0::3]), np.dot(W1, x_ar[:, 1::3]), np.dot(W1, x_ar[:, 2::3]))))
    h1 = a1

    # Layer 2 (Hidden Layer)
    a2 = b2 + np.dot(W2, h1)
    h2 = sigmoid(a2)

    # Layer 3 (Output Layer)
    a3 = b3 + np.dot(W3, h2)
    h3 = softmax(a3)
    y_hat = h3

    return a1, h1, a2, h2, a3, h3, y_hat


# In[9]:


# Back-Propogation Function
def back_propogation(a1, h1, a2, h2, a3, h3, y_hat, x_ar, y_ar, n, reg_lambda, W1, W2, W3, b2, b3, b):
    # Calculating Loss Function
    reg = reg_lambda * (
                np.linalg.norm(W1) + np.linalg.norm(W2) + np.linalg.norm(W3) + np.linalg.norm(b2) + np.linalg.norm(b3))
    L = np.mean(-np.multiply(y_ar, np.log(y_hat)).sum(axis=0)) + reg
    # Layer 3 (Output Layer)
    g = y_hat - y_ar

    d_b3 = (np.mean(g, axis=1)).reshape(250, 1) + reg_lambda * 2 * b3
    d_W3 = np.dot(g, h2.T) / b + reg_lambda * 2 * W3
    g = np.dot(W3.T, g)

    # Layer 2 (Hidden Layer)
    g = g * derive_sigmoid(a2)

    d_b2 = (np.mean(g, axis=1)).reshape(128, 1) + reg_lambda * 2 * b2
    d_W2 = np.dot(g, h1.T) / b + reg_lambda * 2 * W2
    g = np.dot(W2.T, g)

    # Layer 1 (Embedding Layer)
    g = g * a1

    x1 = x_ar[:, 0::3]

    x2 = x_ar[:, 1::3]

    x3 = x_ar[:, 2::3]
    d_W1 = ((np.dot(np.split(g, 3)[0], x1.T) + np.dot(np.split(g, 3)[1], x2.T) + np.dot(np.split(g, 3)[2],
                                                                                        x3.T))) / b + reg_lambda * 2 * W1

    # Updates

    W1 = W1 - n * d_W1
    W2 = W2 - n * d_W2
    W3 = W3 - n * d_W3

    b2 = b2 - n * d_b2
    b3 = b3 - n * d_b3
    return W1, W2, W3, b2, b3, L


# In[ ]:


# Main.py


import numpy as np
from matplotlib import pyplot as plt 
from Network import* # Importing the Network Class

# Importing necessary files
vocab = np.load('data/vocab.npy')
train_inputs = np.load('data/train_inputs.npy')
train_targets = np.load('data/train_targets.npy')

valid_inputs = np.load('data/valid_inputs.npy')
valid_targets = np.load('data/valid_targets.npy')

test_inputs = np.load('data/test_inputs.npy')
test_targets = np.load('data/test_targets.npy')


# In[2]:


# Defining one-hot representations for each dataset 
traini_ar = gen_mbatch(train_inputs)
traint_ar = gen_mbatch(train_targets.reshape(len(train_targets),1))

validi_ar = gen_mbatch(valid_inputs)
validt_ar = gen_mbatch(valid_targets.reshape(len(valid_targets),1))

testi_ar = gen_mbatch(test_inputs)
testt_ar = gen_mbatch(test_targets.reshape(len(test_targets),1))

# Accuracy Calculation Function
def accuracy(x_ar, y_ar, W1, W2, W3, b2, b3):
    return (np.argmax(y_ar, axis=0) == np.argmax(forward_propogation(x_ar, W1, W2, W3, b2, b3)[6], axis=0)).sum() / y_ar.shape[1]
 


# In[3]:


# For loop for epoc and minibatch algorithm
np.random.seed(1234) # Fixin seed number

# Initializing the weight matrices and biases from normal 
W1 = np.random.rand(16,250)
W2 = np.random.rand(128,48)
W3 = np.random.rand(250,128)
b2 = np.random.rand(128,1)
b3 = np.random.rand(250,1)

b = 2048 # Minibatch Size
epoches = 10 # Number of epocs for training
n = 5 # Defining the learning rate
reg_lambda = 0 # Defining regularization parameter (L2 Norm)

Loss = np.empty((0,1))
Train_Acc = np.empty((0,1))
Valid_Acc = np.empty((0,1))


for epoc in range(epoches):
    n = n/(epoc+1)
    mini_batches = gen_minibatch(train_inputs, train_targets,b) # Shuffle the data and generate minibatches
    for j in range(len(mini_batches)):
        x = mini_batches[j][:, [0,1,2]]
        y = mini_batches[j][:, [3]]
        y = y.reshape(len(y),1)
    
        x_ar = gen_mbatch(x)
        y_ar = gen_mbatch(y)
    
        # Forward Propogation
        a1, h1, a2, h2, a3, h3, y_hat = forward_propogation(x_ar, W1, W2, W3, b2, b3)
    
        # Back-Propogation
        W1, W2, W3, b2, b3, L = back_propogation(a1, h1, a2, h2, a3, h3, y_hat, x_ar, y_ar, n, reg_lambda, W1, W2, W3, b2, b3, b)
    
        print(epoc,j,L,np.mean(np.argmax(y_hat, axis=0)))
        Loss = np.append(Loss,L)
    Train_Acc = np.append(Train_Acc,accuracy(traini_ar, traint_ar, W1, W2, W3, b2, b3))
    Valid_Acc = np.append(Valid_Acc,accuracy(validi_ar, validt_ar, W1, W2, W3, b2, b3))
    print(Train_Acc, Valid_Acc)
    
    


# In[5]:


# Accuracy dataset
Results = np.array([range(epoches)]).reshape(epoches,1)
Results = np.append(Results, Train_Acc.reshape(epoches,1),axis=1)
Results = np.append(Results, Valid_Acc.reshape(epoches,1),axis=1)
import pandas as pd
print (pd.DataFrame(Results))


# In[6]:


# Plotting Accuracies
plt.title("Train and Validation Accuracy") 
plt.xlabel("Epoc") 
plt.ylabel("Accuracy") 
plt.plot(Results[:,1], label=" Training")
plt.plot(Results[:,2], label=" Validation")
plt.savefig('accuracy_plot.png')
plt.legend()
plt.show()


# In[7]:


# Plotting Loss Function
plt.title("Loss Function") 
plt.xlabel("Iteration") 
plt.ylabel("Loss") 
plt.plot(Loss)
plt.savefig('Loss_plot.png')
plt.show()


# In[8]:


# Saving the model parameters in .pk format
import pickle
with open('model.pk', 'wb') as f:
    pickle.dump([W1, W2, W3, b2, b3], f)

#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 5 EVALUATION FUNCTION (eval.py)
import numpy as np
import pickle
from Network import* # Importing the Network Class (network.py)
# First, load the learned parameters
with open('model.pk', 'rb') as f:
    W1, W2, W3, b2, b3 = pickle.load(f)


# Loading the test data

test_inputs = np.load('data/test_inputs.npy')
test_targets = np.load('data/test_targets.npy')

testi_ar = gen_mbatch(test_inputs)
testt_ar = gen_mbatch(test_targets.reshape(len(test_targets),1))

# Accuracy Calculation Function
def accuracy(x_ar, y_ar, W1, W2, W3, b2, b3):
    return (np.argmax(y_ar, axis=0) == np.argmax(forward_propogation(x_ar, W1, W2, W3, b2, b3)[6], axis=0)).sum() / y_ar.shape[1]

Test_Accuracy = accuracy(testi_ar, testt_ar, W1, W2, W3, b2, b3) # Test Accuracy
print(Test_Accuracy)


# In[ ]:

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 6 Creating 2D plot using TSNE function (tsne.py)

import numpy as np
import pickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

vocab = np.load('data/vocab.npy') # Loading the vocab
with open('model.pk', 'rb') as f:
    W1, W2, W3, b2, b3 = pickle.load(f)

o_vocab = np.zeros((len(vocab), len(vocab))) # One-hot representation of vocab words
np.fill_diagonal(o_vocab, 1)
data = np.dot(W1,o_vocab).T # learned embeddings


m = TSNE(learning_rate=0.05, random_state = 1234) # TSNE model
tsne_features = m.fit_transform(data)


# In[2]:


# Plotting the embeddings
plt.figure(figsize=(15,12))
X = tsne_features[:,0]
Y = tsne_features[:,1]
plt.scatter(X,Y,s=1,color="white")
plt.title("Scatter Plot of Embeddings",fontsize=15)
for i, label in enumerate(vocab):
    plt.annotate(label, (X[i], Y[i]))
plt.savefig('tsne_plot', dpi=100)

plt.show()


# In[3]:


# Comments on clusters
# 'city of new' , 'life in the' and 'he is the'
from Network import* # Importing the Network Class
inputs = np.array([
    np.array([np.where(vocab == 'city'), np.where(vocab == 'of'),np.where(vocab == 'new')]).flatten(),
    np.array([np.where(vocab == 'life'), np.where(vocab == 'in'),np.where(vocab == 'the')]).flatten(),
    np.array([np.where(vocab == 'he'), np.where(vocab == 'is'),np.where(vocab == 'the')]).flatten()])

predictions = forward_propogation(gen_mbatch(inputs), W1, W2, W3, b2, b3)[6]
print("Predictions",vocab[np.argmax(predictions,axis=0)]) # predicted words




