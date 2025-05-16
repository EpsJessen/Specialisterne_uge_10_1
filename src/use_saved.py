from load_data import get_loaders
from os.path import join
from neural_network import get_nn
import torch
from torch import nn
from main import test_loop, get_hyper_parameters




def main():
    model = get_nn()
    model.load_state_dict(torch.load(join("data", "trained_model.pth")))
    model.eval()
    hyper_parameters = get_hyper_parameters()
    batch_size = hyper_parameters.get("batch_size", 64)
    (train_loader, test_loader) = get_loaders(batch_size)
    loss_fn = nn.CrossEntropyLoss()

    test_loop(test_loader, model, loss_fn)



if __name__ == "__main__":
    main()
