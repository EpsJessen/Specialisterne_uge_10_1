from load_data import get_loaders
from os.path import join
import json
from neural_network import get_nn, NeuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from copy import deepcopy


def get_hyper_parameters():
    parameters_location = join("src", "hyper_parameters.json")
    with open(parameters_location) as parameters_file:
        parameters_dict: dict = json.load(parameters_file)
    return parameters_dict


def train_loop(
    dataloader: DataLoader, model: NeuralNetwork, loss_fn, optimizer, batch_size
):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    hyper_parameters = get_hyper_parameters()
    epochs = hyper_parameters.get("epochs", 5)
    batch_size = hyper_parameters.get("batch_size", 64)
    learning_rate = hyper_parameters.get("learning_rate", 1e-3)

    (train_loader, test_loader) = get_loaders(batch_size)

    model = get_nn()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    state = deepcopy(model.state_dict())
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        lr = learning_rate/((t+1)**0.75)
        print(f"learning rate: {lr:>3e}")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        train_loop(train_loader, model, loss_fn, optimizer, batch_size)
        test_loop(test_loader, model, loss_fn)
    print("Done!")

    model.load_state_dict(state)
    test_loop(test_loader, model, loss_fn)


if __name__ == "__main__":
    main()
