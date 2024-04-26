# Environment Set Up
# Working with MNIST Dataset
# This is the data proparation part

# import necessary libraries

import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define a function to distribute the images at random IID to 10 clients with the 60k images
# So we have 60k/10 = 6k for each cient

def minstIID(dataset, num_users):
    num_images = int(len(dataset)/ num_users)
    users_dict, indexes = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i) # Seed to get the same numbers everytime
        users_dict[i] = set(np.random.choice(indexes, num_images, replace=False))
        indexes = list(set(indexes) - users_dict[i])
    return users_dict

# This is the minst data set in a non iid format
def minstNonIID(dataset, num_users):
    classes, images = 100, 600
    classes_index = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indexes = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indexes_unlabels = np.vstack(indexes, unsorted_labels)
    labels = indexes_unlabels[:, indexes_unlabels[1, :].argsort] 
    indexes = labels[0, :]
    
    for i in range(num_users):
        temp = set(np.random.choice(classes_index, 2, replace = False))
        classes_index = list(set(classes_index)-temp)

        for t in temp:
            users_dict[i] = np.concatenate((
                users_dict[i], indexes[t*images:(t+1)*images]), axis=0)
    return users_dict




# A function to load the dataset with iid type
def load_dataset(num_users, iidtype):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train = True , download = True, transform = transform)
    test_dataset = datasets.MNIST('./data', train = False , download = True, transform = transform)
    train_group, test_group = None, None
    if iidtype == 'iid':
        train_group = minstIID(train_dataset, num_users)
        test_group = minstIID(test_dataset, num_users)
    elif iidtype == 'noniid':
        rain_group = minstNonIID(train_dataset, num_users)
        test_group = minstNonIID(test_dataset, num_users)
    else:
        rain_group = minstNonIIDUnequal(train_dataset, num_users)
        test_group = minstNonIIDUnequal(test_dataset, num_users)        
    return train_dataset, test_dataset, train_group, test_group

# A class for the Federated learning dataset

class FedDataset(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = [int(i) for i in index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        images, label = self.dataset[self.index[item]]
        return torch.tensor(images), torch.tensor(label)
    
# Defined method to get the actual images of the dataset
def getActualImages(dataset, indexes, batch_size):
    return DataLoader(FedDataset(dataset, indexes), batch_size=batch_size, shuffle=True) 

# Example
# import numpy as np

# random_selected_classes = np.random.randint(1,11, size = 10)
# print(random_selected_classes)
# print(sum(random_selected_classes))
# random_selected_classes = np.around(random_selected_classes/ sum(random_selected_classes)*1500)
# print(random_selected_classes)
# random_selected_classes = random_selected_classes.astype(int)
# print(random_selected_classes)
# print(sum(random_selected_classes.astype(int)))

# Checking the implementation in a None iid unequal instance

def minstNonIIDUnequal(dataset, num_users):
    classes, images = 1200, 50
    classes_index = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indexes = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indexes_labels = np.vstack((indexes,labels))
    indexes_labels = indexes_labels[:, indexes_labels[1, :].argsort] 
    indexes = indexes_labels[0, :]

    min_cls_per_client = 1
    max_cls_per_client = 30

    random_selected_classes = np.random.tint(min_cls_per_client, max_cls_per_client+1, size = num_users)
    random_selected_classes = np.around(random_selected_classes/ sum(random_selected_classes) * classes)
    random_selected_classes = random_selected_classes.astype(int)

    if sum(random_selected_classes)>classes:
        temp = set(np.random.choice(classes_index, 1, replace=False))
        classes_index = list(set(classes_index) - temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indexes[t*images:(t+1)*images]), axis=0)
        random_selected_classes = random_selected_classes - 1
    
    for i in range(num_users):
        if len(classes_index) == 0:
            continue
        class_size = random_selected_classes[i]
        if class_size > len(classes_index):
            class_size = len(classes_index)
        
        temp = set(np.random.choice(classes_index, class_size, replace = False))
        classes_index = list(set(classes_index)-temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indexes[t*images:(t+1)*images]), axis=0)

    else:
        for i in range(num_users):
            class_size = random_selected_classes[i]
            temp = set(np.random.choice(classes_index, class_size, replace = False))
            classes_index = list(set(classes_index)-temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indexes[t*images:(t+1)*images]), axis=0)
        
        if len(classes_index) > 0:
            class_size = len(classes_index)
            k = min(users_dict, key = lambda x: len(users_dict.get(x)))
            temp = set(np.random.choice(classes_index, class_size, replace = False))
            classes_index = list(set(classes_index)-temp)
            for t in temp:
                users_dict[k] = np.concatenate((users_dict[k], indexes[t*images:(t+1)*images]), axis=0)
    
    return users_dict
