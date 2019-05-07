from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from network import Net
import utils
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import dataset_menagement
import torchvision.transforms as transforms
import torchvision

# import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)


# Training settings !----------------!

batch_size = 64 # Dimensione del batch

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

epochs = 3 # Numero di epoche di addestramento

transform = transforms.ToTensor()


# End Training Settings !------------!

def train(epoch, data_loader):
    model.train()
    for batch_idx, (data, target, targat_aux) in enumerate(data_loader):
        if cuda:
            data, target, target_aux = Variable(data.cuda()), Variable(target.cuda()), Variable(target_aux.cuda())
        data, target, target_aux = Variable(data), Variable(target), Variable(target_aux)

        optimizer.zero_grad()
        output, output_aux = model(data)

        loss = F.nll_loss(output, target)
        loss_aux = F.nll_loss(output_aux, target_aux)

        loss = loss + loss_aux

        loss.backward()
        loss_aux.backward()

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(data_loader)*batch_size,
                100. * batch_idx / len(data_loader), loss.data.item()))

def test(data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, target_aux in enumerate(data_loader):
        if cuda:
            data, target, target_aux = Variable(data.cuda()), Variable(target.cuda()), Variable(target_aux.cuda())
        data, target, target_aux = Variable(data), Variable(target), Variable(target_aux)
        # data, target = Variable(data, volatile=True), Variable(target) vecchia riga

        output, output_aux = model(data)

        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()

        test_loss += F.nll_loss(output_aux, target_aux, reduction='sum').data.item()

        # Gestire quante ne indovina con le etichette ausiliarie fare nuove predizioni e ritornarle

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct.item() / len(data_loader.dataset)))

    accuracy = 100. * (correct.item() / len(data_loader.dataset))
    # print(accuracy)
    return accuracy


def validate(epoch, data_loader):
    model.eval()
    val_loss, correct = 0, 0
    for batch_idx, (data, target, target_aux) in enumerate(data_loader):
        if cuda:
            data, target, target_aux = Variable(data.cuda()), Variable(target.cuda()), Variable(target_aux.cuda())
        data, target = Variable(data), Variable(target), Variable(target_aux)
        output, output_aux = model(data)

        # sum up batch loss
        val_loss += F.nll_loss(output, target).item()
        val_loss += F.nll_loss(output_aux, target).item()

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(data_loader)*batch_size,
                        100. * batch_idx / (len(data_loader)), val_loss))
    val_loss /= len(data_loader)*batch_size
    accuracy = 100. * correct / (len(data_loader)*batch_size)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, (len(data_loader)*batch_size), accuracy))


# test che prova il dataset menagement e dovrebbe fare i calcoli senza fare la validazione (per ora) analizzando le
# triplette nel database data,target,target_aux
# la parte del validation c'è già bisogna modificare dataset_management per caricare il db
# spezzato in train validation e test
if __name__ == '__main__':

    accuracy, accuracy_validation = [], []

    start = timer()

    trainset = dataset_menagement.AuxDataset(torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True))
    testset = dataset_menagement.AuxDataset(torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True))

    for epoch in range(0, epochs):

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=False,
                                                  num_workers=4)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False,
                                                 num_workers=4)

        train(epoch, data_loader=trainloader)
        accuracy.append(test(data_loader=testloader))




    end = timer()
    print('Execution time in minutes: {:.2f}'.format((end - start) / 60))

    # Stampo grafici

    # x = np.arange(1, epochs + 1)
    # plt.plot(x, accuracy, '-o', label="accuracy")
    # plt.legend()
    # plt.title('not_validate_vs_validate')
    # plt.xlabel('num_epoch')
    # plt.ylabel('accuracy_%')
    # plt.show()
