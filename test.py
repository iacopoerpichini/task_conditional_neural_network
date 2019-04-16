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


#import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)


# Training settings
batch_size = 64

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

epochs = 5

classify_with_classes01 = True;

def train(epoch, data_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(data_loader)*batch_size,
                100. * batch_idx / len(data_loader), loss.data.item()))


def validate(epoch, data_loader):
    # model.train()
    model.eval()
    val_loss, correct = 0, 0
    # for batch_idx, (data, target) in enumerate(data_loader):
    #     data, target = Variable(data), Variable(target)
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = F.nll_loss(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     if batch_idx % 10 == 0:
    #         print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch + 1, batch_idx * len(data), len(data_loader.dataset),
    #             100. * batch_idx / len(data_loader), loss.data.item()))
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = Variable(data), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(data_loader)*batch_size,
                        100. * batch_idx / (len(data_loader)), val_loss))
    val_loss /= len(data_loader)*batch_size
    #qui la len del dataset mi prende 60000 ma il dataloader dovrebbe essere 5000 -RISOLTO
    accuracy = 100. * correct / (len(data_loader)*batch_size)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, (len(data_loader)*batch_size), accuracy))


def test(data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if cuda:
            data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = Variable(data), Variable(target)
        # data, target = Variable(data, volatile=True), Variable(target) vecchia riga

        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum').data.item()
        # test_loss += F.nll_loss(output, target, size_average=False).data.item() vecchia riga

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct.item() / len(data_loader.dataset)))

    accuracy = 100. * (correct.item() / len(data_loader.dataset))
    # print(accuracy)
    return accuracy

# DA FARE UNA FUNZIONE CHE CLASSIFICA CON GLI 0 E 1 E UNA CHE CLASSIFICA CON LE CLASSI 0,1,2,3,...,9

if __name__ == '__main__':

    accuracy, accuracy_validation = [], []

    start = timer()

    for epoch in range(0, epochs):
        train_loader, test_loader = utils.getMNIST(validation=False, batch_size=batch_size)

        if (classify_with_classes01):
            utils.classify_loader(train_loader, 'train')
            utils.classify_loader(test_loader, 'test')

        train(epoch, data_loader=train_loader)
        accuracy.append(test(test_loader))

        # dividere train set 55000:5000 e fare validation set da 5000
        train_loader, validation_loader, test_loader = utils.getMNIST(validation=True, batch_size=batch_size)

        if (classify_with_classes01):
            utils.classify_loader(train_loader, 'train_val')
            utils.classify_loader(validation_loader,'validation')
            utils.classify_loader(test_loader, 'test')

        train(epoch, data_loader=train_loader)
        validate(epoch, data_loader=validation_loader)  # da vedere questa funzione se ha senso
        accuracy_validation.append(test(test_loader))

    end = timer()
    print('Execution time in minutes: {:.2f}'.format((end - start) / 60))

    # Stampo grafici

    x = np.arange(1, epochs + 1)
    plt.plot(x, accuracy, '-o', label="accuracy")
    plt.plot(x, accuracy_validation, '-rs', label="accuracy_validation", linestyle='--')
    plt.legend()
    plt.title('not_validate_vs_validate')
    plt.xlabel('num_epoch')
    plt.ylabel('accuracy_%')
    plt.show()

#test da 5 epoche circa 11 minuti