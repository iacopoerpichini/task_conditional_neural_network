from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

kernel_size = 3

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=2) #stride = 2,3?
        # METTERE UN IF CHE RIPRENDE I DATI DALLA SECONDA RETE QUANDO SARÀ IMPLEMENTATA ?
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 128)
        self.fc2_drop = nn.Dropout(0.2)

        # Aux net
        self.globalAvgPool = nn.AvgPool2d(kernel_size=kernel_size, stride=28)
        # lo stesso qui controllare le grandezze
        self.feat_fc1 = nn.Linear(32,32)
        self.feat_fc2 = nn.Linear(32,64)

        self.aux_fc3 = nn.Linear(64,128)

        self.beta_fc = nn.Linear(64,32) # DOVREBBERO RIENTRARE 32 FEATURES A CONV 2 COME PARAMETRI? no pensare bene come fare per trasformare array in matrice metodo expand as consigliato prof
        self.gamma_fc = nn.Linear(64,32)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        y = F.relu(x) # Mi salvo y dopo la prima convoluzione e dopo il relu prima del max pool per condizionare la rete
        # y = x
        x = self.maxPool2(y)
        # potrei salvarmi y anche qui

        # Aux net
        y = self.globalAvgPool(y)
        # da controllare con il professore se ha senso
        y_size = y.size(0) # modifico il tensore per darlo ai due livelli fc
        y = y.view(y_size,-1)
        y = self.feat_fc1(y)
        y = self.feat_fc2(y)

        beta = self.beta_fc(y)
        gamma = self.gamma_fc(y)
        # capire come passare beta e gamma a un custom layer che li rimanda in input a conv 2
        # qui passare come parametro il condizionamento
        # x = parameter_conditioning(x, beta, gamma) # qui x è img 13*13*32 credo vedere come gestire beta e gamma

        x = F.relu(self.maxPool2(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = self.fc2_drop(x)


        # adesso in zeta c'è l'output per classificare, prima dovrei estrarre beta e gamma per condizionare la rete
        z = self.aux_fc3(y)
        # z = z.reshape(64,32,128) # lui que aveva un tensore [64,32,1,128] cosi lo fa [64,32,128] non capisco da dove arriva il 32 # dovrei aver risolto
        # dovrebbe essere [64,128]



        loss, task_loss = F.log_softmax(x, dim=1), F.log_softmax(z, dim=1)
        return loss, task_loss

# Funzione che prende in input output conv 1,bias(beta), scaling parameters(gamma) e restituisce una conv con i
# parametri condizionati
def parameter_conditioning(output_conv, bias, scaling_parameters):
    # DOVREI USARE QUESTO PER IL BIAS?
    # stdv = 1. / math.sqrt(self.weight.size(1))
    # self.weight.data.uniform_(-stdv, stdv)
    # if self.bias is not None:
    #     self.bias.data.uniform_(-stdv, stdv)
    #
    # gamma = ScaleLayer(gamma)
    #
    # [(1-gamma)*CONV1(IMG) + beta]
    # CAPIRE COME FARE LA MOLTIPLICAZIONE E PASSARLA IN INPUT AL LAYER CONV2
    # le dimensioni sono [64,32] per bias e scaling_parameters e [64,32,13,13] per output_conv
    print('vediamo cosa c\'é nei parametri')
    return (bias+((1-scaling_parameters)*output_conv))


    # def extract_conv1(self, x, avg_pool = True): #ha senso per estrarre l'output dopo la prima convoluzione?
    #     x = x.view(-1, 14*14) #28*28 dimezzato dopo conv 1 ?
    #     if(avg_pool):
    #         self.avgPool = nn.AvgPool2d(kernel_size=kernel_size, stride=14) #kernel = 28 stride = 1 global avg poll global max pool mettere lo stride = alla grandezza della finestra
    #         x = F.relu(self.avgPool(self.conv1(x)))
    #     else :
    #         self.maxPool = nn.MaxPool2d(kernel_size=kernel_size, stride=14)
    #         x = F.relu(self.maxPool(self.conv1(x)))
    #     return x

    # def extract_features(data_loader):
    #     model.eval()
    #     conv1 = []
    #     labels = []
    #     for data, target in data_loader:
    #         if cuda:
    #             data, target = data.cuda(), target.cuda()
    #         data, target = Variable(data), Variable(target)
    #         conv1.append(model.extract_conv1(data))
    #         labels.append(target)
    #     return (torch.cat(conv1), torch.cat(labels)) #con cat concateno i tensori

# DOVREI USARE QUESTA CLASSE PER SCALARE I PARAMETRI Y
# [(1-Y)*CONV1(IMG) + Bias]

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale


if __name__ == '__main__':
    model = Net()

    print(model)