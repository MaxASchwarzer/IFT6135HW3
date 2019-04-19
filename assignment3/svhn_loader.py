import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


def sanitize_mat(filename):
    mat = sio.loadmat(filename)

    data = mat["X"]
    label = mat["y"].astype(np.int64).squeeze()

    # Replace class label 10 with class label 0 to comply with pytorch.
    # If we have C classes, the class label range from 0 to C-1 and not
    # 1 to C
    np.place(label, label == 10, 0)
    # PyTorch/Torch uses data of shape: Batch x Channel x Height x Width
    # SVHN data is in shape: Height x Width x Channel x Batch.
    data = np.transpose(data, (3, 2, 0, 1))

    sio.savemat(filename, {"X": data, "y": label})


def create_train_val_split(split=0.20):
    """
    Create a train and validation split and save into separate files

    Args:
        split (int): The percentage of data to be in the validation set
                        from the test set
    """
    train_mat = sio.loadmat('train_32x32.mat')

    data = train_mat["X"]
    label = train_mat["y"]
    # label = train_mat["y"].astype(np.int64).squeeze()
    # # Replace class label 10 with class label 0 to comply
    # np.place(label, svhn_label == 10, 0)
    # data = np.transpose(data, (3, 2, 0, 1))

    # Shuffle
    combined = list(zip(data, label))
    random.shuffle(combined)
    data[:], label[:] = zip(*combined)

    # Split
    split_idx = int(data.shape[0]*split)
    train_data = data[split_idx:]
    train_label = label[split_idx:]
    val_data = data[:split_idx]
    val_label = label[:split_idx]

    sio.savemat("train_split_32x32.mat", {"X": train_data, "y": train_label})
    sio.savemat("val_split_32x32.mat", {"X": val_data, "y": val_label})


class TestDataset(Dataset):
    """SVHN Test dataset"""

    def __init__(self, mat_file_loc, transform=None):
        """
        Args:
            mat_file_loc (string): Path to the mat file containing test data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test_mat = sio.loadmat(mat_file_loc)
        self.data = self.test_mat["X"]
        self.label = self.train_mat["y"]
        # self.data = np.transpose(self.test_mat["X"], (3, 2, 0, 1))
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]


class TrainDataset(Dataset):
    """SVHN Train dataset"""

    def __init__(self, mat_file_loc, transform=None):
        """
        Args:
            mat_file_loc (string): Path to the mat file containing train data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_mat = sio.loadmat(mat_file_loc)
        self.data = self.train_mat["X"]
        self.label = self.train_mat["y"]
        # self.data = np.transpose(self.train_mat["X"], (3, 2, 0, 1))
        # self.label = self.train_mat["y"].astype(np.int64).squeeze()
        # self.label = np.place(self.label, self.label == 10, 0)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return {"data": self.transform(self.data[idx]), "label": self.label[idx]}
        return self.transform(self.data[idx]), self.label[idx]


class ValidDataset(Dataset):
    """SVHN Validation dataset"""

    def __init__(self, mat_file_loc, transform=None):
        """
        Args:
            mat_file_loc (string): Path to the mat file containing validation data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.val_mat = sio.loadmat(mat_file_loc)
        self.data = self.val_mat["X"]
        self.label = self.val_mat["y"]
        # self.data = np.transpose(self.val_mat["X"], (3, 2, 0, 1))
        # self.label = self.val_mat["y"].astype(np.int64).squeeze()
        # self.label = np.place(self.label, self.label == 10, 0)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return {"data": self.transform(self.data[idx]), "label": self.label[idx]}
        return self.transform(self.data[idx]), self.label[idx]


# Run sanitization only once
sanitize_mat("train_32x32.mat")
sanitize_mat("test_32x32.mat")

# Split train into train and val. Run this once
create_train_val_split()

test_batch_size = 128
train_batch_size = 128
valid_batch_size = 128

test_transform = transforms.Compose([transforms.ToTensor()])
train_transform = transforms.Compose([transforms.ToTensor()])
valid_transform = transforms.Compose([transforms.ToTensor()])

test_dataset = TestDataset(
    mat_file_loc="test_32x32.mat", transform=test_transform)
train_dataset = TrainDataset(
    mat_file_loc="train_split_32x32.mat", transform=train_transform)
valid_dataset = ValidDataset(
    mat_file_loc="valid_split_32x32.mat", valid_transform=valid_transform)


test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                         shuffle=False, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                          shuffle=True, num_workers=4)

valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size,
                          shuffle=False, num_workers=4)
