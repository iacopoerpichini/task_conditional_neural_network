import sys
import torch
import torchvision
from torch.utils.data import sampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import bisect
import warnings

from torch._utils import _accumulate
from torch import randperm

dict = { # Classificazione ausiliaria per ora a caso
        0: 0,
        1: 1,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 0,
        8: 0,
        9: 0
    }

class AuxDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, dataset):
        self._ds = dataset

    def __getitem__(self, index):
        x, y = self._ds[index]
        # print(x)
        # print(y)
        y_aux = dict[y]

        # return (x,y),y_aux
        return x,y,y_aux

        raise NotImplementedError

    def __len__(self):
        return self._ds
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])



class TensorDataset(AuxDataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)



class ConcatDataset(AuxDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    # modificare il get item per far tornare la tripletta img,target,target_aux
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



class Subset(AuxDataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)



def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    print('Proviamo a fare una gestione del dataset')

    transform = transforms.ToTensor()
    train_size = 60000

    trainset = AuxDataset(torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True))
    testset = AuxDataset(torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True))

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=False,
                                              num_workers=4, sampler=ChunkSampler(train_size, 0))

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False,
                                             num_workers=4)

    # Fare vari test a console

    print(trainset._ds.__len__())

    img = trainset.__getitem__(9)[0].numpy().reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.show()
    print(trainset.__getitem__(9)[0]) # mi restituisce l'item (img, classificazione)
    # con [0] solo l'img se metto [1] il valore target con [2] il target aux

    img = trainset.__getitem__(234)[0].numpy().reshape(28, 28)
    plt.imshow(img)
    plt.show()
    print(trainset.__getitem__(234))  # mi restituisce l'item [(img, classificazione), y_aux]

     # se faccio trainloader.dataset[0] mi restituisce la tripletta
    # ma se faccio trainloader.dataset._ds.__getitem__(0) non mi restituisce l'ultimo y
    #credo che si debba modificare il get item

    # trainset.__getItem__(1) ok ma trainset._ds.__getItem__()