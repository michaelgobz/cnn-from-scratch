import torch
import torch.nn as nn
import pytest


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # -> 16x224x224
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x8x8

            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x4x4

            # Since we are using BatchNorm and data augmentation,
            # we can go deeper than before and add one more conv layer
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 128x14x14

            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x14x14
            # Add batch normalization (BatchNorm2d) here
            # YOUR CODE HERE
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 256x7x7

            nn.Flatten(),  # -> 1x256x7x7

            nn.Linear(1*256*7*7, 6500),  # -> 12500
            # Add batch normalization (BatchNorm1d, NOT BatchNorm2d) here
            # YOUR CODE HERE
            nn.Dropout(dropout),
            nn.BatchNorm1d(6500),
            nn.ReLU(),
            nn.Linear(6500, 2000),  # -> 2000
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Linear(2000, 500),  # -> 500
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_classes),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()
    print(images.shape)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

#%%
