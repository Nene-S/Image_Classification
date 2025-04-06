import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import config
import load_data
from resnetModel import ResNet, ResidualBlock
from model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(10)

def train(model, epochs=15, batch_size=10, learning_rate=0.001):
    """
    Trains a CNN model using the given dataset.

    Args:
        epochs (int, optional): Number of training epochs. Default is 15.
        batch_size (int, optional): Batch size for training and validation. Default is 10.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.

    Returns:
        tuple:
            - model (torch.nn.Module): The trained CNN model.
            - hist (tuple of lists): Training and validation history containing:
                - loss_hist_train (list): Training loss per epoch.
                - loss_hist_valid (list): Validation loss per epoch.
                - accuracy_hist_train (list): Training accuracy per epoch.
                - accuracy_hist_valid (list): Validation accuracy per epoch.
    """

    train_dataset, valid_dataset, _ = load_data.load_dataset()
    
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model.to(device) 
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    loss_hist_train = [0] * epochs
    accuracy_hist_train = [0] * epochs
    loss_hist_valid = [0] * epochs
    accuracy_hist_valid = [0] * epochs

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_func(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()

            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f"Epoch {epoch+1} train accuracy: {accuracy_hist_train[epoch]:.4f} validation accuracy: {accuracy_hist_valid[epoch]:.4f}")
        scheduler.step()
    return model, (loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid)

def learning_curve(hist, path):
    """
    Plots and saves the learning curve of the model.
    Args:
        hist (tuple of lists): The training history containing:
            - loss_hist_train (list): Training loss per epoch.
            - loss_hist_valid (list): Validation loss per epoch.
            - accuracy_hist_train (list): Training accuracy per epoch.
            - accuracy_hist_valid (list): Validation accuracy per epoch.
        path (str): The file path to save the learning curve plot.
    """
    
    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(13, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], "-o", label="Train loss")
    ax.plot(x_arr, hist[1], "--<", label="Validation loss")
    ax.legend(fontsize=12)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Loss", size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], "-o", label="Train accuracy")
    ax.plot(x_arr, hist[3], "--<", label="Validation accuracy")
    ax.legend(fontsize=12)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Accuracy", size=15)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def main():
    """
    Main function to train models, save them, and generate learning curve plots.
    """
    models = [(CNNModel().to(device), config.cnn_model_path, config.cnn_curve_path),
        (ResNet(ResidualBlock, n_classes=3, n_blocks_list=[2, 2, 2, 2], 
                out_channels_list=[64, 128, 256, 512], num_channels=3).to(device), 
         config.resnet_model_path, config.resnet_curve_path) ]

    for model, model_path, curve_path in models:
        trained_model, history = train(model, epochs=23, batch_size=20)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        learning_curve(history, curve_path)

if __name__ == "__main__":
    main()
