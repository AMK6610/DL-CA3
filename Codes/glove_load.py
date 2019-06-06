import bcolz
import numpy as np
import pickle

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir='../glove/42B/42B.300.dat', mode='w')

with open('../glove/42B/glove.42B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((1917494, 300)), rootdir='../glove/42B/42B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open('../glove/42B/42B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('../glove/42B/42B.300_idx.pkl', 'wb'))