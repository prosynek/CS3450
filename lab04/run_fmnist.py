# ## Lab04: Fashion-MNIST
#
# **Paige Rosynek**
#
# - 04.13.2023

import os
import torch
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 1
# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

# With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")

def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_folded_training_data():
    """
    This method introduces a single non-linear fold into the sort of data created by create_linear_training_data. Be sure to REMOVE the final softmax layer before testing on this data!
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0).float() - 0.5)
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_square():
    """
    This is a square example in which the challenge is to determine
    if the points are inside or outside of a point in 2d space.
    insideness is true if the points are inside the square.
    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    win_x = [2,2,3,3]
    win_y = [1,2,2,1]
    win = torch.tensor([win_x,win_y],dtype=torch.float32)
    win_rot = torch.cat((win[:,1:],win[:,0:1]),axis=1)
    t = win_rot - win # edges tangent along side of poly
    rotation = torch.tensor([[0, 1],[-1,0]],dtype=torch.float32)
    normal = torch.matmul(rotation,t) # rotation @ t # normal vectors to each side of poly
        # torch.matmul(rotation,t) # Same thing

    points = torch.rand((2,2000),dtype = torch.float32)
    points = 4*points

    vectors = points[:,np.newaxis,:] - win[:,:,np.newaxis] # reshape to fill origin
    insideness = (normal[:,:,np.newaxis] * vectors).sum(axis=0)
    insideness = insideness.T
    insideness = insideness > 0
    insideness = insideness.all(axis=1)
    return points, insideness


def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a numpy array where columns are training samples and
             y is a numpy array where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only to a numpy array"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.detach().numpy(), y=y.detach().numpy())
    return x, y

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename,'a') as file:
                print(str(datetime.datetime.now())+": ",message,file=file)

if __name__ == '__main__':        
    with Timer('Total time'):
        # TODO: Select your datasource.
        dataset = 'Fashion-MNIST'
        x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)
        print(f'x_train = {x_train.shape}')
        
        #--------- hyperparameters ---------
        EPOCHS = 20   
        BATCH_SIZE = 20
        NUM_BATCHES = x_train.shape[1] // BATCH_SIZE
        learning_rate = 0.01
        reg =  0.001
        epsilon = 1e-8
        #------------------------------------
        
        # TODO: Build your network. 
        W = torch.randn(256, 784) * 0.1                  # weights : input -> hidden
        W.requires_grad = True
        b_1 = torch.zeros(256, 1, requires_grad=True)     

        M = torch.randn(10, 256) * 0.1                  # weights : hidden -> output
        M.requires_grad = True
        b_2 = torch.zeros(10, 1, requires_grad=True)
        
        train_accuracy = []
    
        # TODO: Train your network.
        with Timer('Training time'):
            # pass # Replace with your code to train
            #---------- training loop ----------
            for epoch in range(EPOCHS):
                total_loss = 0.0
                total_accuracy = 0.0
                
                print(f'=======================================================================')
                print(f'Epoch: {epoch + 1} / {EPOCHS}')
                print(f'=======================================================================')
                
                for i in range(0, x_train.shape[1], BATCH_SIZE): # for i in range(NUM_BATCHES + 1):
                    x_batch = x_train[:, i:i+BATCH_SIZE]
                    y_batch = y_train[:, i:i+BATCH_SIZE]

                    #---------- forward pass ----------
                    z = torch.matmul(W, x_batch) + b_1                 # input -> hidden
                    h = z * (z > 0)                                    # relu
                    o = torch.matmul(M, h) + b_2                       # hidden -> output 
                    exp_o = torch.exp(o - torch.max(o, dim=0).values)
                    y_hat = exp_o / torch.sum(exp_o, dim=0)            # activation - softmax 

                    #---------- cross entropy loss ----------
                    k = torch.sum(-1 * y_batch * torch.log(y_hat + epsilon), dim=1)   # epsilon avoids log(0)  
                    L = torch.mean(k)
                    
                    #---------- regularization ----------             
                    s = (reg / 2) * ((torch.sum(W**2)) + (torch.sum(M**2)))         # regularization term
                    
                    #---------- objective function ----------
                    J = L + s 
                    
                    #---------- accuracy ----------
                    y_pred = torch.argmax(y_hat, dim=0)   
                    y_true = torch.argmax(y_batch, dim=0) 
                    total_accuracy += torch.sum((y_pred == y_true).float())
                    total_loss += L

                    # backpropagation
                    J.backward()

                    # update weights - gradient descent
                    # no_grad : updates weights without changing / tracking the changes in gradients
                    with torch.no_grad():
                        W -= learning_rate * W.grad
                        b_1 -= learning_rate * b_1.grad
                        M -= learning_rate * M.grad
                        b_2 -= learning_rate * b_2.grad

                    # clear the gradients
                    W.grad.zero_()
                    b_1.grad.zero_()
                    M.grad.zero_()
                    b_2.grad.zero_()

                    
                    # print('-------------------------------------------------------------------------------------------')
                    # print(f'Batch: {i // BATCH_SIZE}/{NUM_BATCHES}\tLoss: {L}')
                    
                epoch_accuracy = (total_accuracy / x_train.shape[1]) * 100
                train_accuracy.append(epoch_accuracy)
                
                # print accuracy every epoch
                print(f'ACCURACY:\t{epoch_accuracy} %')
                print(f'LOSS:\t\t{total_loss / NUM_BATCHES}')
                print(f'=======================================================================')
                print()
            
            
        # PLOT TRAINING CURVE
        plt.plot(range(1, EPOCHS + 1), train_accuracy)
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy (%)')
        plt.title('Training Accuracy vs Epoch')
        plt.show()
        
        # TODO: Sanity-check the output of your network.
        # You can optionally compute the error on this test data:
        x_test, y_test = load_dataset_flattened(train=False, dataset=dataset)
        
        #---------- forward pass ----------
        z = torch.matmul(W, x_test) + b_1                 # input -> hidden
        h = z * (z > 0)                                    # relu
        o = torch.matmul(M, h) + b_2                       # hidden -> output 
        exp_o = torch.exp(o)
        y_hat = exp_o / torch.sum(exp_o, dim=0)            # activation - softmax 

        #---------- cross entropy loss ----------
        k = torch.sum(-1 * y_test * torch.log(y_hat + epsilon), dim=1)   # epsilon avoids log(0)  
        L = torch.mean(k)

        #---------- regularization ----------             
        s = (reg / 2) * ((torch.sum(W**2)) + (torch.sum(M**2)))         # regularization term

        #---------- objective function ----------
        J = L + s 

        #---------- accuracy ----------
        y_pred = torch.argmax(y_hat, dim=0)   
        y_true = torch.argmax(y_test, dim=0) 
        test_accuracy = (y_pred == y_true).sum().item()
        
        
        print(f'TEST ACCURACY:\t{(test_accuracy / x_test.shape[1]) * 100} %')
        print(f'=======================================================================')
        print()
    
        # Report on GPU memory used for this script:
        peak_bytes_allocated = torch.cuda.memory_stats()['active_bytes.all.peak']
        print(f"Peak GPU memory allocated: {peak_bytes_allocated} Bytes")

    pass # You may wish to keep this line as a point to place a debugging breakpoint.


