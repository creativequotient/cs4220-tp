import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as D
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc
import time
import os
import sys

from cnn import CNN
from resnet import make_resnet
from utils import CustomDataset, custom_collate_fn, graph_losses


if __name__ == '__main__':
    dataset_1, labels_1 = np.load('../data/images/x_images_arrays.npz'), np.load('../data/images/y_labels.npz')
    dataset_2, labels_2 = np.load('../data/images/x_images_arrays_2.npz'), np.load('../data/images/y_labels_2.npz')

    data_info = {"dataset_1": (dataset_1, labels_1), "dataset_2": (dataset_2, labels_2)}

    images = []
    labels = []

    for dataset in data_info:
        x, y = data_info[dataset]
        images.append(x['arr_0'])
        labels.append(y['arr_0'])

    images, labels = np.concatenate(images, axis=0).astype(np.float32) / 255.0, np.concatenate(labels, axis=0)

    labels = np.array(list(map(lambda x: int(x[0]), labels)))

    nucleoplasm_idx = np.argwhere(labels == 0)
    mitochondria_idx = np.argwhere(labels == 14)
    cytosol_idx = np.argwhere(labels == 16)
    golgi_idx = np.argwhere(labels == 7)
    nuclear_speckles_idx = np.argwhere(labels == 4)
    plasma_membrane_idx = np.argwhere(labels == 13)
    centrosome_idx = np.argwhere(labels == 12)

    # class 1
    nucleoplasm_imgs = images[nucleoplasm_idx]
    nucleoplasm_labels = np.zeros((len(nucleoplasm_idx), 1))

    # class 2
    mitochondria_imgs = images[mitochondria_idx]
    mitochondria_labels = np.ones((len(mitochondria_idx), 1))

    # class 3
    cytosol_imgs = images[cytosol_idx]
    cytosol_labels = np.ones((len(cytosol_idx), 1))
    cytosol_labels[:] = 2

    # class 4
    golgi_imgs = images[golgi_idx]
    golgi_labels = np.ones((len(golgi_idx), 1))
    golgi_labels[:] = 3

    # class 5
    nuclear_speckles_imgs = images[nuclear_speckles_idx]
    nuclear_speckles_labels = np.ones((len(nuclear_speckles_idx), 1))
    nuclear_speckles_labels[:] = 4

    # class 6
    plasma_membrane_imgs = images[plasma_membrane_idx]
    plasma_membrane_labels = np.ones((len(plasma_membrane_idx), 1))
    plasma_membrane_labels[:] = 5

    # class 7
    centrosome_imgs = images[centrosome_idx]
    centrosome_labels = np.ones((len(centrosome_idx), 1))
    centrosome_labels[:] = 6

    images = np.concatenate([nucleoplasm_imgs, mitochondria_imgs, cytosol_imgs, golgi_imgs, nuclear_speckles_imgs, plasma_membrane_imgs, centrosome_imgs], axis=0)
    images = np.squeeze(images, axis=1)
    labels = np.concatenate([nucleoplasm_labels, mitochondria_labels, cytosol_labels, golgi_labels, nuclear_speckles_labels, plasma_membrane_labels, centrosome_labels], axis=0)

    del dataset_1
    del dataset_2
    gc.collect()

    print(f'images shape: {images.shape} dtype: {images.dtype}\nlabels shape: {labels.shape} dtype: {labels.dtype}')

    # Constants
    epochs = 15
    validate_every = 5

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, random_state=42, shuffle=True, train_size=0.75, test_size=0.25)

    # Definte transforms
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # Init DataLoader
    loaders = {}

    train_dataset = CustomDataset(X_train, y_train, transform)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=6, drop_last=True)

    eval_dataset = CustomDataset(X_test, y_test, transform)
    eval_sampler = torch.utils.data.RandomSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, drop_last=True)

    loaders['train'] = train_loader
    loaders['eval'] = eval_loader

    # Initialize model
    # model = make_resnet(num_classes=7).to(device='cuda')
    model = CNN(num_classes=7).to(device='cuda')
    learning_rate = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    loaders = loaders
    criterion = criterion
    best_loss = 1000.0
    running_losses = {'train': [], 'eval': []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        phases = ['train']
        if epoch % validate_every == 0:
            phases.append('eval')

        for phase in phases:
            model.eval() if phase == 'eval' else model.train()
            gc.collect() # prevent OOM problems

            print("Epoch {}/{} Phase {}".format(epoch, epochs, phase))
            for idx, (imgs, labels) in enumerate(tqdm(loaders[phase])):
                # print(f'Phase: {phase}, current step: {idx}')
                imgs, labels = imgs.float().to(device='cuda'), labels.float().to(device='cuda')
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels.long())
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_losses[phase].append(loss.cpu().detach().numpy())

            mean_loss = np.array(running_losses[phase]).mean()
            if phase == 'eval':
                # print("Eval running loss: ", running_losses['eval'])
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    # xm.save(model.state_dict(), 'model_best.pth')
            print(epoch, mean_loss, best_loss, (time.time()-start_time)/60**1)


    accuracies = []
    for idx, (imgs, labels) in enumerate(tqdm(loaders['eval'])):
        imgs, labels = imgs.float().to(device='cuda'), labels
        outputs = model(imgs)
        selected_class = torch.argmax(outputs, dim=-1).detach().cpu().numpy()
        accuracies.append(accuracy_score(selected_class, labels))
    print(f'avg accuracy: {sum(accuracies) / len(accuracies)}')
