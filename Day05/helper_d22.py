# Shin-Han Shiu
# 3/18/2024
# Helper functions for Day 22 exercises

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_subsets(data_dir, train_valid_size=200):
  '''Get training, validation, and test data
  Args:
    data_dir (str or Path): the directory to store the data
    train_valid_size (int): the number of images for training and validation
  Return:
    X_train (torch.Tensor): training images, defined by train_valid_size
    X_valid (torch.Tensor): validation images, defined by train_valid_size
    X_test (torch.Tensor): test images, all 6000 images per class
    y_train (torch.Tensor): training labels
    y_valid (torch.Tensor): validation labels
    y_test (torch.Tensor): test labels
    class_names (list): the names of the classes
  '''

  fashion_train_data =  datasets.FashionMNIST(root=data_dir, train=True, 
                download=True, transform=transforms.ToTensor())
  fashion_test_data  =  datasets.FashionMNIST(root=data_dir, train=False, 
                download=True, transform=None)

  class_names = fashion_train_data.classes

  # features and labels for the full training dataset
  X_train_all = fashion_train_data.data
  y_train_all = fashion_train_data.targets

  # populate selected images
  X_train_list = []
  X_valid_list = []
  X_test_list  = []

  # populate selected labels
  y_train_list = []
  y_valid_list = []
  y_test_list  = []

  # go through each class
  for class_i in range(10):
    # get labels and features of class_i
    y_train_i = y_train_all[y_train_all == class_i]
    X_train_i = X_train_all[y_train_all == class_i]

    # append the first train_valid_size to training set and the next 
    # train_valid_size to validation set, also normalize the features
    X_train_list.append(X_train_i[:train_valid_size]/255)
    X_valid_list.append(X_train_i[train_valid_size:train_valid_size*2]/255)
    y_train_list.append(y_train_i[:train_valid_size])
    y_valid_list.append(y_train_i[train_valid_size:train_valid_size*2])

    # deal with testing data
    X_test_list.append(X_train_i/255)
    y_test_list.append(y_train_i)

  # concatenate images in lists into tensor of shape (2000, ...)
  X_train = torch.cat(X_train_list, dim=0)
  X_valid = torch.cat(X_valid_list, dim=0)
  y_train = torch.cat(y_train_list, dim=0)
  y_valid = torch.cat(y_valid_list, dim=0)
  X_test  = torch.cat(X_test_list, dim=0)
  y_test  = torch.cat(y_test_list, dim=0)

  return X_train, X_valid, X_test, y_train, y_valid, y_test, class_names

def plot_images(images, targets, classes, img_indices):
  '''Plot images from dataset
  Args:
    images (np.array): array of images, shape (n_images, height, width)
    targets (np.array): array of targets, shape (n_images,)
    classes (list): list of class names, in the order of class labels
    img_indices (list): list of image indices to plot
  '''

  n_cols = 5
  n_rows = len(img_indices) // n_cols + 1 

  plt.figure(figsize=(8, 2*n_rows))
  plt.subplots_adjust(hspace=0.5)

  for idx, img_idx in enumerate(img_indices):
    # comment
    X = images[img_idx]

    # comment
    y = targets[img_idx]

    ax = plt.subplot(n_rows, n_cols, idx+1)
    ax.imshow(X, cmap="binary")
    ax.set_title(f"idx {img_idx}\n{classes[y]} ({y})")
    ax.axis('off')

  plt.show()

def plot_train_valid_scores(epoch_loss, epoch_f1):
  '''Plot the training and validation losses and F1 over epochs
  Args:
    epoch_loss (list): the loss after each epoch
    epoch_f1 (list): the f1 score after each epoch
  '''
  
  epochs = [i[0] for i in epoch_loss]

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
  for idx, scores in enumerate([epoch_loss, epoch_f1]):
    score_type = "loss" if idx == 0 else "f1"
    train_scores = [i[1].item() for i in scores]
    valid_scores = [i[2].item() for i in scores]

    ax[idx].plot(epochs, train_scores, label=f'Train {score_type}', c='b', 
                  alpha=0.3)
    ax[idx].plot(epochs, valid_scores, label=f'Valid {score_type}', c='r', 
                  alpha=0.3)
    ax[idx].set_xlabel('Epochs')
    ax[idx].set_ylabel(score_type)
    ax[idx].grid(True)

    x_text = 0
    scores_all = train_scores + valid_scores
    score_range = np.max(scores_all)-np.min(scores_all)
    y_text = (score_range)/2+np.min(scores_all)
    ax[idx].text(x_text, y_text, 
                 f"Final {score_type} train: {train_scores[-1]:.2f}",)
    ax[idx].text(x_text, y_text-score_range*0.1,
                 f"Final {score_type} valid: {valid_scores[-1]:.2f}")
    ax[idx].legend()
  plt.tight_layout()
  plt.show()