import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def load_training_data() -> Dataset:
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    return training_data


def load_test_data() -> Dataset:
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    return test_data


def get_labels_map() -> dict[int, str]:
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }


def show_sample(training_data, labels_map) -> None:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title([labels_map[label]])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def load_data() -> tuple[Dataset, Dataset]:
    training_data = load_training_data()
    test_data = load_test_data()
    return training_data, test_data


def get_loaders(batch_size) -> tuple[DataLoader, DataLoader]:
    training_data, test_data = load_data()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def main():
    training_data, test_data = load_data()
    labels_map = get_labels_map()
    show_sample(training_data, labels_map)


if __name__ == "__main__":
    main()
