import sys
import torch
import torchvision
from torch.utils.data import sampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# SETTINGS MESSI COME PARAMETRI PASSABILI ALLE FUNZIONI ?
# batch_size = 64
#     If using CUDA, num_workers should be set to 1 and pin_memory to True. trovato on line
# num_workers = 4  # number of workers threads ?


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


def getMNIST(validation=False, batch_size=64, num_workers=4):

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])


    train_size = 60000
    if validation:
        train_size = 55000
        validation_size = 5000

    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, sampler=ChunkSampler(train_size, 0))
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    if validation:
        validationloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers,
                                                       sampler=ChunkSampler(validation_size, train_size))
        return trainloader, validationloader, testloader

    return trainloader, testloader


# test per vedere se splitta effettivamente il dataset
def testSplitData(batch_size=64):
    train_loader, validation_loader, test_loader = getMNIST(validation=True)
    print('con validation')
    count = 0
    for i, data in enumerate(train_loader, 0):
        count += 1
    print('len train loader:{}'.format(count * batch_size))
    count = 0
    for i, data in enumerate(validation_loader, 0):
        count += 1
    print('len validation loader:{}'.format(count * batch_size))
    count = 0
    for i, data in enumerate(test_loader, 0):
        count += 1
    print('len test loader:{}'.format(count * batch_size))

    print('senza validation')
    train_loader, test_loader = getMNIST(validation=False)
    print('len train loader:{}'.format(len(train_loader.dataset)))
    print('len test loader:{}'.format(len(test_loader.dataset)))


# stampo x img di un dataloader
def testStampa(dataloader, num_img):
    # img = dataloader.dataset[0][0].numpy().reshape(28, 28)
    # plt.imshow(img)

    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))

    for i in range(0, num_img):
        plt.subplot(1, num_img, i + 1)
        img = dataloader.dataset[i][0].numpy().reshape(28, 28)
        plt.imshow(img, cmap="gray") #senza cmap stampa a colori?
    plt.show()


if __name__ == '__main__':  # test di tutte le funzioni implementate

    testSplitData()

    num_img = 5

    train_loader, test_loader = getMNIST(validation=False)
    testStampa(dataloader=train_loader, num_img=num_img)
    testStampa(dataloader=test_loader, num_img=num_img)

    train_loader, validation_loader, test_loader = getMNIST(validation=True)
    testStampa(dataloader=train_loader, num_img=num_img)
    testStampa(dataloader=validation_loader, num_img=num_img)
    testStampa(dataloader=test_loader, num_img=num_img)
