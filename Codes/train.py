import torch
from model import Network
import numpy as np
import pickle
from torch.utils import data
import torch.nn as nn
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import random
import matplotlib.pyplot as plt

image_directory = '../data/nyu_depth_images/'
data_directory = '../data/questions/'
glove6b_directory = '../glove/6B/'
glove42b_directory = '../glove/42B/'
glove_directory = '../glove/'
data_file = 'data.pkl'
glove6b_data_file = 'glove6b_data.pkl'
glove42b_data_file = 'glove42b_data.pkl'
glove6b_random_data_file = 'glove6b_random_data.pkl'
glove42b_random_data_file = 'glove42b_random_data.pkl'
result_directory = '../img/'

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')

# parameters
batch_size = 32
validation_train = 5
validation_size = 2494
neg_sample_count = 5
num_epochs = 30

class Dataset(data.Dataset):
    def __init__(self, X, label, lengths, transform=None):
        self.X = X
        self.label = np.array(label)
        self.transform = transform
        self.lengths = lengths
        self.img = pickle.load(open(image_directory + 'resnet18_' + data_file, 'rb'))
        for key, value in self.img.items():
            self.img[key] = value.to(cpu_device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return self.X[index], self.label[index]
        
        # print(img[self.label[index][1]])
        # print(self.X[index], self.label[index][0])
        # self.img = self.img.to(cpu_device)
        # print(self.X[index].shape, self.label[index][0].shape)
        # , self.img[self.label[index][1].to(cuda_device)]
        # print(len(self.X[index]))
        return self.X[index] , self.label[index][0], self.img[self.label[index][1]], self.lengths[index]

def preprocess_data(array):
    for i in range(len(array)):
        array[i] = torch.tensor(array[i])
    lengths = [item.size()[0] for item in array]
    array = pad_sequence(array, batch_first=True)
    return array, lengths



def main():
    partition = pickle.load(open(data_directory + glove6b_data_file, 'rb'))
    # print(np.array(partition['train_data']).shape, np.array(partition['train_label']).shape, np.array(partition['validation_data']).shape, np.array(partition['validation_label']).shape)
    # print(partition['train_label'][4], partition['train_label'][1], partition['train_label'][0], partition['train_label'][3], np.array(partition['train_data']).shape)
    # print(np.array(partition['train_data'])[5].shape)

    train_params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 4}
        
    validation_params = {'batch_size': validation_size,
          'shuffle': False,
          'num_workers': 4}

    partition['train_data'], train_lengths = preprocess_data(partition['train_data'])
    partition['validation_data'], validation_lengths = preprocess_data(partition['validation_data'])
    print(partition['train_data'].size(), len(train_lengths), train_lengths[-1])


    training_set = Dataset(partition['train_data'], partition['train_label'], train_lengths)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = Dataset(partition['validation_data'], partition['validation_label'], validation_lengths)
    validation_generator = data.DataLoader(validation_set, **validation_params)

    randoms = pickle.load(open(glove_directory + glove6b_random_data_file, 'rb'))
    print(randoms.shape)


    loss_function = F.cross_entropy
    # loss_function = F.cross_entropy
    model = Network().double().to(cuda_device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=0.005)

    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    validation_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        i = 0
        epoch_loss = []
        total_test_loss = 0.0
        model.zero_grad()
        for x, y, img, lengths in training_generator:
            # print('image[0] in train:', img[0:6])
            # print('INPUT[0] in train:', x[0])
            # print(x.shape, y.shape, img.shape, lengths.shape)
            # print(x[list(lengths).index(max(list(lengths)))])

            neg_samples = torch.zeros((x.shape[0], neg_sample_count, 300))
            neg_samples_idx = [random.sample(range(lengths[i]), neg_sample_count) for i in range(x.shape[0])]
            for k, list_index in enumerate(neg_samples_idx):
                for l, index in enumerate(list_index):
                    neg_samples[k][l] = x[k][index]
            # print(neg_samples.size())   # 32 * 5 * 300
            x = x.transpose(0, 1)
            t = random.sample(list(randoms), 26 * x.shape[1])
            t = torch.Tensor(t)
            t = t.view(x.shape[1], 26, -1)
            # print(t.size()) # 32 * 26 * 300
            # print(y.shape)  # 32 * 1 * 300
            # 26 * 1 * 300
            # 5 * 1 * 300
            # 32 * 1 * 300
            # out: 32 * 32 * 300
            result = torch.cat((torch.cat((y.unsqueeze(1).double(), t.double()), dim=1), neg_samples.double()), dim=1)
            # print(result.size())  # 32 * 32 * 300

            # print(torch.eq(y[0].float(), result[0][0])) # check

            # x = torch.stack(random.sample(range(lengths[0]), 10))
            # print(x.size())

            
            x, img, lengths, result = x.double().to(cuda_device), img.double().to(cuda_device), lengths.double().to(cuda_device), result.double().to(cuda_device)
            y_pred = model(x, img, lengths, result)
            # print(y.shape, y_pred.shape)
            # print('y_pred:', y_pred, y_pred.size(), y_pred[0].sum())
            loss = loss_function(y_pred, torch.zeros((x.shape[1])).long().to(cuda_device))
            epoch_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print('grad:', model.linear2.weight.grad)
            
            print('[%d, %d] Train loss = %.5f' %
                (epoch + 1, i + batch_size, loss.item()))
            i += batch_size 
        train_loss[epoch] = np.array(epoch_loss).mean()
        print('[%d] Train loss = %.5f' %
                (epoch + 1, np.array(epoch_loss).mean()))

        epoch_validation_loss = []
        for x, y, img, lengths in validation_generator:
            neg_samples = torch.zeros((x.shape[0], neg_sample_count, 300))
            neg_samples_idx = [random.sample(range(lengths[i]), neg_sample_count) for i in range(x.shape[0])]
            for k, list_index in enumerate(neg_samples_idx):
                for l, index in enumerate(list_index):
                    neg_samples[k][l] = x[k][index]
            x = x.transpose(0, 1)
            t = random.sample(list(randoms), 26 * x.shape[1])
            t = torch.Tensor(t)
            t = t.view(x.shape[1], 26, -1)
            result = torch.cat((torch.cat((y.unsqueeze(1).double(), t.double()), dim=1), neg_samples.double()), dim=1)
            x, img, lengths, result = x.double().to(cuda_device), img.double().to(cuda_device), lengths.double().to(cuda_device), result.double().to(cuda_device)
            y_pred = model(x, img, lengths, result)
            loss = loss_function(y_pred, torch.zeros((x.shape[1])).long().to(cuda_device))
            epoch_validation_loss.append(loss.item())
        print('[%d] Validation loss = %.5f' %
                (epoch + 1, np.array(epoch_validation_loss).mean()))
        validation_loss[epoch] = np.array(epoch_validation_loss).mean()


        # print(torch.cat((y, y), dim=1).size())

    

    plt.plot([i + 1 for i in range(num_epochs)], [t for t in train_loss], marker='o')
    plt.plot([i + 1 for i in range(num_epochs)], [t for t in validation_loss], marker='o')
    plt.title('Changes in loss function')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    #plt.figure()
    plt.savefig(result_directory + 'loss-9.jpg')
    plt.show()

    pickle.dump({'train': train_loss, 'validation': validation_loss}, open(result_directory + '9.pkl', 'wb'))

    # plt.plot([i + 1 for i in range(num_epochs)], accuracy, marker='o', color='red')
    # plt.title('Changes in accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('accuracy')
    # #plt.figure()
    # plt.savefig(result_directory + 'acc-4.jpg')
    # plt.show()


if __name__ == '__main__':
    main()