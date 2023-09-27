# Paige Rosynek
# CS 3450 021
# Lab 08 - Training & Testing w/ From-Scratch Network
# 05.11.2023

import torch
import numpy as np # For loading cached datasets
import matplotlib.pyplot as plt
# import torchvision # For loading initial datasets
#                    # Commented out because Spring 2023 this is failing to load
#                    # in the conda-cs3450 environment
import time
import warnings
import os.path

import network as nn
import layers

# warnings.filterwarnings('ignore')  # If you see warnings that you know you can ignore, it can be useful to enable this.

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
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
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
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
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
    normal = rotation @ t # normal vectors to each side of poly
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
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
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
            """Like ToTensor, only redefined by us for 'historical reasons'"""

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
        np.savez(path, x=x.numpy(), y=y.numpy())
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


# Training loop -- fashion-MNIST
# def main_linear():
if __name__ == '__main__':
    # Once you start this code, comment out the method name and uncomment the
    # "if __name__ == '__main__' line above to make this a main block.
    # The code in this section should NOT be in a helper method.
    #
    # In particular, your client code that uses your classes to stitch together a specific network 
    # _should_ be here, and not in a helper method.  This will give you access
    # to the layers of your network for debugging purposes.

    with Timer('Total time'):    
        #============== select dataset ============== 
        dataset = 'Fashion-MNIST'
        # x_train, y_train = create_linear_training_data()
        # x_test, y_test = create_linear_training_data()
        # x_train, y_train = create_folded_training_data()
        x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)
        x_test, y_test = load_dataset_flattened(train=False, dataset=dataset)
        print(f'x_train = {x_train.shape}')
        print(f'x_test = {x_test.shape}')
        
        #============== hyperparameters ============== 
        EPOCHS = 10
        BATCH_SIZE = 1
        NUM_BATCHES = x_train.shape[1] // BATCH_SIZE
        learning_rate = 0.01
        r_lambda =  0.001

        print(f'============ HYPERPARAMETERS ============')
        print(f'Epochs: {EPOCHS}\nBatch size: {BATCH_SIZE}\nLearning rate = {learning_rate}\nRegularization: {r_lambda}\n')
        #============== build network ============== 

        # test network - linear data, 0 regularization
        network = nn.Network()
        x = layers.Input((784,1), False)
        W = layers.Input((256,784), True)
        b = layers.Input((256,1), True)
        z = layers.Linear(x, W, b)
        h = layers.ReLU(z)
        M = layers.Input((10,256), True)
        c = layers.Input((10,1), True)
        o = layers.Linear(h, M, c)
        y_true = layers.Input(o.output.shape, False)
        L = layers.Softmax(o, y_true)
       # L = layers.MSELoss(y_true, o)
        s1 = layers.Regularization(r_lambda, W)
        s2 = layers.Regularization(r_lambda, M)
        s = layers.Sum(s1, s2)
        J = layers.Sum(L, s)                    # objective function

        # init model parameters
        W.randomize()
        b.randomize()
        M.randomize()
        c.randomize()

        network.set_input(x)
        #network.add(x)
        network.add(W)
        network.add(b)
        network.add(z)
        network.add(h)
        network.add(M)
        network.add(c)
        network.add(o)
        network.add(y_true)
        network.add(L)
        network.add(s1)
        network.add(s2)
        network.add(s)
        network.add(J)


        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        #============== training loop ==============
        with Timer('Training time'):
            for epoch in range(EPOCHS):
                train_epoch_loss = 0.0
                train_epoch_accuracy = 0.0
                for i in range(NUM_BATCHES):
                    x_batch = x_train[:, i:i+BATCH_SIZE]
                    y_batch = y_train[:, i:i+BATCH_SIZE]
        
                    x.set(x_batch)
                    y_true.set(y_batch)

                    # forward pass
                    network.forward()

                    # accuracy
                    pred = torch.argmax(o.output, dim=0) 
                    true = torch.argmax(y_batch, dim=0) 
                    train_epoch_accuracy += torch.sum((pred == true).float())

                    # loss - check this
                    train_epoch_loss += network.output

                    # backpropagation
                    network.backward()
                    network.step(learning_rate)

                
                # calculate the accuracy & loss on the training set
                train_epoch_accuracy = (train_epoch_accuracy / x_train.shape[1]) * 100
                train_epoch_loss = (train_epoch_loss / x_train.shape[1]) * 100
                train_accuracy.append(train_epoch_accuracy)
                train_loss.append(train_epoch_loss)


                # 1 epoch - testing
                test_epoch_loss = 0.0
                test_epoch_accuracy = 0.0
                for i in range(x_test.shape[1]):
                    x_test_batch = x_test[:, i].reshape((x_test.shape[0], 1))
                    y_test_batch = y_test[:, i].reshape((y_test.shape[0], 1))

                    x.set(x_test_batch)
                    y_true.set(y_test_batch)

                    # forward pass
                    network.forward()

                    # accuracy
                    pred = torch.argmax(o.output, dim=0) 
                    true = torch.argmax(y_test_batch, dim=0) 
                    test_epoch_accuracy += torch.sum((pred == true).float())

                    # loss - check this
                    test_epoch_loss += network.output


                # calculate the accuracy & loss on the test set
                test_epoch_accuracy = (test_epoch_accuracy / x_test.shape[1]) * 100
                test_epoch_loss = (test_epoch_loss / x_test.shape[1]) * 100
                test_accuracy.append(test_epoch_accuracy)
                test_loss.append(test_epoch_loss)

                # output performance after each epoch
                print(f'===============================================================================================================================================')
                print(f'EPOCH {epoch + 1}')
                print(f'Train Accuracy: {train_epoch_accuracy:<11} %\tTrain Loss: {train_epoch_loss:<11}\tTest Accuracy: {test_epoch_accuracy:<11} %\tTest Loss: {test_epoch_loss:<11}'.ljust(130))
                #print(f'=====================================================================================================')
                
            print()


        #----------PLOT RESULTS----------
        # plot test-train curves
        epochs = np.arange(1, EPOCHS + 1)
        print(f'train_loss = {train_loss}')
        print(f'test_loss = {test_loss}')
        print(f'train_accuracy = {train_accuracy}')
        print(f'test_accuracy = {test_accuracy}')
        
        # loss
        plt.plot(epochs, torch.tensor(train_loss).cpu(), label='Loss - Train')
        plt.plot(epochs, torch.tensor(test_loss).cpu(), label='Loss - Test')
        plt.title('Fashion-MNIST Loss v. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'./cs3450/lab08/loss_{EPOCHS}epochs_{learning_rate}step_{r_lambda}_reg.png')
        plt.clf()


        # accuracy
        plt.plot(epochs, torch.tensor(train_accuracy).cpu(), label='Accuracy - Train')
        plt.plot(epochs, torch.tensor(test_accuracy).cpu(), label='Accuracy - Test')
        plt.title('Fashion-MNIST Accuracy v. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(f'./cs3450/lab08/accuracy_{EPOCHS}epochs_{learning_rate}step_{r_lambda}_reg.png')
        plt.clf()


        plt.plot(epochs, torch.tensor(train_loss).cpu(), label='Loss - Train')
        plt.plot(epochs, torch.tensor(test_loss).cpu(), label='Loss - Test')
        plt.plot(epochs, torch.tensor(train_accuracy).cpu(), label='Accuracy - Train')
        plt.plot(epochs, torch.tensor(test_accuracy).cpu(), label='Accuracy - Test')
        plt.title('Fashion-MNIST Training & Testing Results')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'./cs3450/lab08/{EPOCHS}epochs_{learning_rate}step_{r_lambda}_reg.png')
        plt.clf()

        # TODO: Sanity-check the output of your network.
        # Compute the error on this test data:
        # x_test, y_test = create_linear_training_data()
        
        # x_test, y_test = create_folded_training_data()
        # x_test, y_test = load_dataset_flattened(train=False, dataset=dataset)
    
        # Report on GPU memory used for this script:
        peak_bytes_allocated = torch.cuda.memory_stats()['active_bytes.all.peak']
        print(f"Peak GPU memory allocated: {peak_bytes_allocated} Bytes")

    pass # You may wish to keep this line as a point to place a debugging breakpoint.
