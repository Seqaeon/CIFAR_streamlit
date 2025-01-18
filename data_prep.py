import numpy as np
import pandas as pd
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from os import listdir
from os.path import isfile, join

dataset_list = ['CIFAR-10', 'CIFAR-100']
# import from MNIST dataset on AWS using the instructions here: https://stackoverflow.com/a/40693405/4147579
#def unpickle():
#    file_path = "./mnist.pkl.gz"
#    import gzip
#
#    f = gzip.open(file_path, "rb")
#    if sys.version_info < (3,):
#        # TODO: pyright is telling me this is unreachable, check later
#        data = pickle.load(f)
#    else:
#        data = pickle.load(f, encoding="bytes")
#    f.close()
#    return data







def process_labels(labels):
    label_to_binary = np.zeros([100, 7], dtype="int8")
    for i in np.arange(100):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 7)), dtype=int)

    # Changing the labels from 0-9 int to binary for our weightless neural state machine
    labels_z = np.zeros([labels.size, 7])
    for i in np.arange(labels.size):
        labels_z[i] = label_to_binary[labels[i]]
    return labels_z


def process_data(dataset='CIFAR-10'):
    transform = transforms.Compose([
                                    #transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.1736,), (0.3248,)),
                                    ])
#    if dataset == 'MNIST':
#
#        data = unpickle()
#        (MN_TRAIN, MN_TRAIN_labels), (MN_TEST, MN_TEST_labels) = data
#        MN_TRAIN_Z = process_labels(MN_TRAIN_labels)
#        MN_TEST_Z = process_labels(MN_TEST_labels)
#        label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if dataset == "CIFAR-100":
        train_complete = datasets.CIFAR100(root='./data', train=True,download=True,transform=transform)
        test_complete = datasets.CIFAR100(root='./data', train=False,download=True,transform=transform)
        (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z) = (train_complete.data,process_labels(np.array(train_complete.targets))) , (test_complete.data, process_labels(np.array(test_complete.targets)))
        label_names = train_complete.classes
    elif dataset == "CIFAR-10":
        train_complete = datasets.CIFAR10(root='./data', train=True,download=True,transform=transform)
        test_complete = datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
        (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z) = (train_complete.data,process_labels(np.array(train_complete.targets))) , (test_complete.data, process_labels(np.array(test_complete.targets)))
        label_names = train_complete.classes

    else:
        raise ValueError("Invalid dataset option selected. Please choose from 'CIFAR-10', 'CIFAR-100'.")

    return (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z), label_names

#def choose_data(dataset='CIFAR-10'):
#    MN_TRAIN_list, MN_TRAIN_Z_list = [], []
#    MN_TEST_list, MN_TEST_Z_list = [], []
#
#    for data in dataset:
#        (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z), _ = process_data(data)
#        MN_TRAIN_list.append(MN_TRAIN)
#        MN_TRAIN_Z_list.append(MN_TRAIN_Z)
#        #print(MN_TRAIN_Z.shape)
#        MN_TEST_list.append(MN_TEST)
#        MN_TEST_Z_list.append(MN_TEST_Z)
#
#    MN_TRAIN = np.vstack(MN_TRAIN_list)
#    MN_TRAIN_Z = np.vstack(MN_TRAIN_Z_list)
#    MN_TEST = np.vstack(MN_TEST_list)
#    MN_TEST_Z = np.vstack(MN_TEST_Z_list)
#
#    return (MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z)


def random_sample(num_samples, samples, labels):
    index = np.random.choice(samples.shape[0], num_samples, replace=False)
    return samples[index], labels[index]

def random_sample_same(num_samples, samples, labels, seed=42):
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(samples.shape[0], size=num_samples, replace=False)
    return samples[index], labels[index]

def down_sample_item(x, down=200):
    f = np.vectorize(lambda x, down: 1 if x >= down else 0)
    return f(x, down)

def bitmap_to_binary(image):
    if image.ndim == 1:
        # Handle 1D array
        return np.array([np.array(list(format(pixel, '08b')), dtype=np.uint8) for pixel in image])
    # elif image.ndim == 2:
    #     # Handle 2D array
    #     return np.array([[np.array(list(format(pixel, '08b')), dtype=np.uint8) for pixel in row] for row in image])
    else:
        # Handle 3D or higher-dimensional arrays
        return np.array([bitmap_to_binary(sub_array) for sub_array in image])


#def get_font_data(filename):
#    df = pd.read_excel(filename)
#    # cut off dataframe at end of valid columns
#    df = df.iloc[:, :145]
#    arr = df.to_numpy()
#    # spreadsheet has two layers of digits
#    # this moves them into the same layer
#    arr = np.append(arr[:28], arr[30:58], axis=1)
#    arr = np.delete(arr, [28 + 29 * i for i in range(10)], axis=1)
#    ret_arr = np.zeros((10, 28, 28), dtype=np.uint8)
#    for i in range(10):
#        for r in range(28):
#            for c in range(28):
#                # TODO: python is complaining about some cells being cast from NaN,
#                #       but it seems to work anyways. Find a fix for this
#
#                # Multiplying by 255 so it downsamples correctly
#                # print("file: {}, digit: {}, r:, {}, c: {}".format(filename, i, r, c))
#                ret_arr[i][r][c] = 255 * arr[r][c + 28 * i]
#    # Convert labels
#    label_to_binary = np.zeros([10, 6], dtype="int8")
#    for i in np.arange(10):
#        label_to_binary[i] = np.array(list(np.binary_repr(i, 6)), dtype=int)
#
#    return (ret_arr, np.array([label_to_binary[i] for i in range(10)]))


#def get_all_fonts():
#    folder = "./Fonts/"
#    items = listdir(folder)
#    fonts = {}
#    for item in items:
#        if not isfile(join(folder, item)):
#            continue
#        file = join(folder, item)
#        fonts[item.removesuffix(".xlsx")] = get_font_data(file)
#    return fonts


#def select_training_fonts(fonts):
#    inputs = []
#    outputs = []
#    for f in fonts:
#        if f == "MNIST":
#            continue
#        inputs.extend(FONTS[f][0])
#        outputs.extend(FONTS[f][1])
#    return np.array(inputs), np.array(outputs)


#(MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z) = choose_data(dataset='CIFAR-10')
(MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z), label_names = process_data('CIFAR-10')
#a,b, label_names = process_data(dataset='CIFAR-10')
#FONTS = get_all_fonts()
