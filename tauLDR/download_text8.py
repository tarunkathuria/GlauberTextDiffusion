import urllib.request
import shutil
import zipfile
import os
import pickle
from collections import Counter
import numpy as np

import string 

url = 'http://mattmahoney.net/dc/text8.zip'
filename = 'text8.zip'
train_size = 90000000
val_size = 95000000

if not os.path.isfile(filename):
    print('Downloading text8 dataset...')

    with urllib.request.urlopen(url) as response, \
        open(filename, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)

rawdata = zipfile.ZipFile(filename).read('text8').decode('utf-8')

train_split = rawdata[:train_size]
valid_split = rawdata[train_size:val_size]
test_split = rawdata[val_size:]

vocab = Counter()

print('Constructing dictionary...')

for word in train_split:
    vocab[word] += 1

wordmap = {char: id for id, char in enumerate(string.ascii_lowercase)}
wordmap[" "] = 26

# vocab_cut = {k: v for k, v in vocab.items() if v > 10}
# vocab_sorted = sorted(vocab_cut.items(), key=lambda x: x[1], reverse=True)
# wordmap = {k: id + 1 for id, (k, _) in enumerate(vocab_sorted)}

def save_pickle(split, wordmap, filename):
    data = []

    for word in split:
        try:
            data.append(wordmap[word])

        except KeyError:
            data.append(26)

    data = np.array(data)
    data_cut = data[:data.shape[0] // (256 * 20) * 20 * 256]

    # data_next = data_cut.copy()
    # data_next[:-1] = data_cut[1:]
    # data_next[-1] = data_cut[0]

    input = data_cut.reshape(-1, 20)
    # input = np.array_split(data_cut.reshape((256, -1)),
    #                        data_cut.shape[0] / 256 / 20, axis=1)
    # label = np.array_split(data_next.reshape((256, -1)),
    #                        data_next.shape[0] / 256 / 20, axis=1)

    # print(input.shape)
    with open(filename, "wb") as f:
        np.save(f, input)

#     with open(filename, 'wb') as f:
#         pickle.dump({'input': input, 'label': label, 'worddic': wordmap}, f)

print('Writing train split...')
save_pickle(train_split, wordmap, 'datasets/text8/train.npy')

print('Writing valid split...')
save_pickle(valid_split, wordmap, 'datasets/text8/valid.npy')

print("done")
