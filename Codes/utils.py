import os
import numpy as np
import pickle
from torch.utils import data
from PIL import Image
import bcolz
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import random

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

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


partition = {
    'test_data': [],
    'train_data': [],
    'validation_data': [],
    'test_label': [],
    'train_label': [],
    'validation_label': [],
    'data': [],
    'label': []
}

# parameters
batch_size = 32
validation_train = 5
validation_size = 2494


def load_questions_data(file_address, file_type):
    j = 0
    with open(file_address + '.txt') as file:
        whole_file = file.read().split('\n')
        for i in range(0, len(whole_file), 2):
            p = file_type
            # print(whole_file[i], whole_file[i + 1])
            if file_type == 'train' and j % validation_train == 0:
                p = 'validation'
            j += 1
            partition[p + '_data'].append(whole_file[i])
            partition[p + '_label'].append(whole_file[i + 1])
            partition['data'].append(whole_file[i])
            partition['label'].append(whole_file[i + 1])

def prepare_questions_data():
    load_questions_data(data_directory + 'train_test', 'train')
    pickle.dump(partition,  open(data_directory + data_file, 'wb'))

def glove(name, address):
    vectors = bcolz.open(address + name +'.300.dat')[:]
    words = pickle.load(open(address + name +'.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(address + name +'.300_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove


class Old_Dataset(data.Dataset):
    def __init__(self, X, label, transform=None):
        self.X = np.array(X)
        self.label = np.array(label)
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # return self.X[index], self.label[index]
        x = self.X[index]
        sentence = x.split()
        img_name = sentence[-2]
        sentence[-2] = 'image'
        img = np.array(Image.open(image_directory + img_name + '.png'))
        img = torch.from_numpy(np.rollaxis(img, 2))
        img = img.float()
        return sentence , self.label[index], img_name

# class Dataset(data.Dataset):
#     def __init__(self, X, label, lengths, transform=None):
#         self.X = X
#         self.label = np.array(label)
#         self.transform = transform
#         self.lengths = lengths
#         self.img = pickle.load(open(image_directory + 'resnet18_' + data_file, 'rb'))
#         for key, value in self.img.items():
#             self.img[key] = value.to(cpu_device)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, index):
#         # return self.X[index], self.label[index]
        
#         # print(img[self.label[index][1]])
#         # print(self.X[index], self.label[index][0])
#         # self.img = self.img.to(cpu_device)
#         # print(self.X[index].shape, self.label[index][0].shape)
#         # , self.img[self.label[index][1].to(cuda_device)]
#         # print(len(self.X[index]))
#         return self.X[index] , self.label[index][0], self.img[self.label[index][1]], lengths
        


# for root, dir, files in os.walk(directory + subdir):
#   i = 0
#   j = 0
#   for file in files:
#     if i % train_test == 0:
#       partition['test'].append(file.split('.')[0])
#     else:
#       j += 1  
#       if j % test_validation == 0:
#         partition['validation'].append(file.split('.')[0])
#       else:
#         partition['train'].append(file.split('.')[0])
#     i += 1

def prepare_images_data(data_generator):
    data = {}
    model = models.densenet121(pretrained=True).to(cuda_device)
    model.fc = nn.Sequential()
    for x, y in data_generator:
        print(np.array(x).shape, np.array(x[1]).shape, np.array(x[2]).shape)
        img = x[1].to(cuda_device)
        with torch.no_grad():
            out = model(img)
        for i, item in enumerate(x[2]):
            data[item] = out[i]
        # dump_tensors()
        # torch.cuda.empty_cache()
        # gc.collect()
        print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())
    pickle.dump(data, open(image_directory + 'densenet121_' + data_file, 'wb'))

def prepare_glove6b_data(training_generator, validation_generator):
    glove6b = glove('6B', glove6b_directory)
    means = pickle.load(open(glove_directory + 'mean.pkl', 'rb'))
    data = {
        'train_data': [],
        'train_label': [],
        'validation_data': [],
        'validation_label': [],
    }
    counter = 0
    for l in [(training_generator, 'train'), (validation_generator, 'validation')]:
        for x, y, z in l[0]:
            # print(x)
            x = np.array(x)
            x = np.rollaxis(x, 1)
            # print(x.shape)
            # print(x)
            for i, sentence in enumerate(x):
                # print(sentence)
                after_glove_sentence = np.zeros((sentence.shape[0], 300))
                # print(after_glove_sentence.shape)
                for j, item in enumerate(sentence):
                    if item not in glove6b.keys():
                        after_glove_sentence[j] = means['6B']
                    else:
                        after_glove_sentence[j] = glove6b[item]
                data[l[1] + '_data'].append(after_glove_sentence)
                # print(sentence, after_glove_sentence, after_glove_sentence.shape)
                # print(after_glove_sentence[10])
            for i, sentence in enumerate(y):
                # print(y)
                answer = y[i].replace(' ', '').split(',')[0]
                answer =  answer.split('_')
                for j, w in enumerate(answer):
                    if w not in glove6b.keys():
                        answer[j] = means['6B']
                    else:
                        answer[j] = glove6b[w]
                # print(answer)
                answer = np.mean(np.array(answer), axis=0)
                data[l[1] + '_label'].append((answer, z[i]))
            # print(data['train_label'], np.array(data['train_data']).shape)
                # print(sentence, answer, answer.shape)
            print(counter)
            counter += 1
        # print(data['train_label'][4], data['train_label'][1], data['train_label'][0], data['train_label'][3], np.array(data['train_data']).shape  )
        pickle.dump(data, open(data_directory + glove6b_data_file, 'wb'))   
            # break

def prepare_glove42b_data(file_address, file_type):
    k = 0
    glove42b = glove('42B', glove42b_directory)
    means = pickle.load(open(glove_directory + 'mean.pkl', 'rb'))
    data = {
        'train_data': [],
        'train_label': [],
        'validation_data': [],
        'validation_label': [],
    }
    with open(file_address + '.txt') as file:
        whole_file = file.read().split('\n')
        for i in range(0, len(whole_file), 2):
            p = file_type
            # print(whole_file[i], whole_file[i + 1])
            if file_type == 'train' and k % validation_train == 0:
                p = 'validation'
            print(k)
            k += 1
            x = whole_file[i]
            y = whole_file[i + 1]
            partition[p + '_label'].append(whole_file[i + 1])
            sentence = x.split()
            img_name = sentence[-2]
            sentence[-2] = 'image'

            after_glove_sentence = np.zeros((len(sentence), 300))
            # print(after_glove_sentence.shape)
            for j, item in enumerate(sentence):
                if item not in glove42b.keys():
                    after_glove_sentence[j] = means['42B']
                else:
                    after_glove_sentence[j] = glove42b[item]
            data[p + '_data'].append(after_glove_sentence)
            # print(sentence, after_glove_sentence, after_glove_sentence.shape)
            # print(after_glove_sentence[10])
            
            # print(y)
            answer = y.replace(' ', '').split(',')[0]
            answer =  answer.split('_')
            for j, w in enumerate(answer):
                if w not in glove42b.keys():
                    answer[j] = means['42B']
                else:
                    answer[j] = glove42b[w]
            # print(answer)
            answer = np.mean(np.array(answer), axis=0)
            data[p + '_label'].append((answer, img_name))
    pickle.dump(data, open(data_directory + glove42b_data_file, 'wb'))



def glove_mean():
    glove6b = glove('6B', glove6b_directory)
    glove42b = glove('42B', glove42b_directory)
    arr1 = np.array(list(glove6b.values()))
    arr2 = np.array(list(glove42b.values()))
    arr1 = np.mean(arr1, axis=0)
    arr2 = np.mean(arr2, axis=0)
    data = {
        '6B': arr1,
        '42B': arr2
    }
    pickle.dump(data, open(glove_directory + 'mean.pkl', 'wb'))

def preprocess_data(array):
    for i in range(len(array)):
        array[i] = torch.tensor(array[i])
    lengths = [item.size()[0] for item in array]
    array = pad_sequence(array, batch_first=True)
    return array, lengths

def n_random_glove(n):
    glove42b = glove('42B', glove42b_directory)
    values = random.sample(list(glove42b.values()), n)
    values = np.array(values)
    print(values.shape)
    pickle.dump(values, open(glove_directory + glove42b_random_data_file, 'wb'))




def main():

    # prepare_questions_data()

    partition = pickle.load(open(data_directory + data_file, 'rb'))
    # print(np.array(partition['train_data']).shape, np.array(partition['train_label']).shape, np.array(partition['validation_data']).shape, np.array(partition['validation_label']).shape, np.array(partition['data']).shape, np.array(partition['label']).shape)

    # partition = pickle.load(open(data_directory + glove6b_data_file, 'rb'))
    # print(np.array(partition['train_data']).shape, np.array(partition['train_label']).shape, np.array(partition['validation_data']).shape, np.array(partition['validation_label']).shape)
    # print(partition['train_label'][4], partition['train_label'][1], partition['train_label'][0], partition['train_label'][3], np.array(partition['train_data']).shape)
    # print(np.array(partition['train_data'])[5].shape)
    
    
    # train_params = {'batch_size': batch_size,
    #       'shuffle': False,
    #       'num_workers': 4}
        
    # validation_params = {'batch_size': validation_size,
    #       'shuffle': False,
    #       'num_workers': 4}

    # validation_params = {'batch_size': 1,
    #         'shuffle': False,
    #         'num_workers': 4}



    # for i in range(len(partition['train_data'])):
    #     partition['train_data'][i] = torch.tensor(partition['train_data'][i])
    #     print(partition['train_data'][i].size()[0])
    # for i in range(len(partition['validation_data'])):
    #     partition['validation_data'][i] = torch.tensor(partition['validation_data'][i])
    # partition['train_data'] = pad_sequence(partition['train_data'], batch_first=True)
    # partition['validation_data'] = pad_sequence(partition['validation_data'], batch_first=True)
    # partition['train_data'], train_lengths = preprocess_data(partition['train_data'])
    # partition['validation_data'], validation_lengths = preprocess_data(partition['validation_data'])
    # print(partition['train_data'].size(), len(train_lengths), train_lengths[-1])


    # training_set = Dataset(partition['train_data'], partition['train_label'])
    # training_generator = data.DataLoader(training_set, **train_params)

    # validation_set = Dataset(partition['validation_data'], partition['validation_label'])
    # validation_generator = data.DataLoader(validation_set, **validation_params)


    # for x, y, img, lengths in training_generator:
    #     print(x.shape, y.shape, img.shape)
    #     break

    
    # x = torch.tensor(partition['train_data'])
    # print(x.shape)
    # print(pad_sequence(x).size())


    # data_set = Dataset(partition['data'], partition['label'])
    # data_generator = data.DataLoader(data_set, **train_params)

    # prepare_glove6b_data(training_generator, validation_generator)

    # prepare_glove42b_data(data_directory + 'train_test', 'train')
    
    # glove_mean()
    # partition = pickle.load(open(glove_directory + 'mean.pkl', 'rb'))
    # print(partition['6B'].shape, partition['42B'].shape)

    # prepare_images_data(data_generator)

    # partition = pickle.load(open(image_directory + 'resnet34_' + data_file, 'rb'))
    # partition2 = pickle.load(open(image_directory + 'resnet18_' + data_file, 'rb'))
    # partition3 = pickle.load(open(image_directory + 'densenet121_' + data_file, 'rb'))
    # print(partition['image1'].shape, partition2['image1'].shape, partition3['image1'].shape)


    # n random gloves:
    # n_random_glove(100000)
    # randoms = pickle.load(open(glove_directory + glove42b_random_data_file, 'rb'))
    # print(randoms.shape)
    # x = random.sample(randoms, 100)
    # x = torch.from_numpy(x)
    # print(x.size())

    # test = pickle.load(open('../img/' + '2.pkl', 'rb'))
    # print(test)

    # vectors = glove('42B', glove42b_directory)
    # print(vectors['kitchen'].shape)

if __name__ == "__main__":
    main()