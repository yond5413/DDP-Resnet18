#DistrubutedDataParallel (DDP) 
## make neural network
'''
Helpful Links 
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
###################################
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
###################################
### new for lab4
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
###################################
import os
import argparse
###################################
import time
from tqdm import tqdm
###################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding, c7 = False):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size, stride,padding,bias = False)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size, stride=1 ,padding=padding, bias = False)
        self.relu  = nn.ReLU(out_channels)
        self.batchNorm  = nn.BatchNorm2d(out_channels)
        self.c7 = c7
        if stride != 1:
            self.down_sample = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1), stride=stride, padding=0, bias = False)
        else:
            self.down_sample =None       
    def forward(self,x):
        identity = x
        out1 = self.conv1(x)
        if self.c7== False:
            f = self.relu(self.batchNorm(out1))
        else:
            f = self.relu(out1)
        #################
        f = self.conv2(f)
        #################
        if self.down_sample:    
            identity = self.down_sample(identity)
            ## should I apply relu and batch-norm here?
        #print(f"size of tensors f: {f.size()}, identity: {identity.size()}, out1: {out1.size()}")
        h = f+identity
        ###
        if self.c7 == False: 
            h  =self.batchNorm(h)#self.batchNorm(self.relu(h))
        ret = self.relu(h) #self.relu(ret)#self.relu(h)
        return ret
##############################
class ResNet(nn.Module,):
    def __init__(self,c7 = False):
        super(ResNet,self).__init__()
        ### 2 basicblocks per sub group
        ###
        ''' 
        input->[64]
        1st block: [64->64],[64,64]
        2nd block: [64->128],[128,128] [input,output]
        3rd block: [128->256],[256,256]
        4th block: [256->,512],[512,512]
        '''
        self.c7 = c7
        #(3,3) -> 3x3 
        # stride may only impact the input layer for residuals?
        self.input_layer = nn.Conv2d(in_channels = 3,out_channels=64,kernel_size=(3,3), stride = 1,padding=1)#ConvBlock()
        ### has default parmas ^
        #print("Resnet-18 model init")
        self.block1 =    ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1,c7 = c7)
        self.block1_b =  ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1,c7 = c7)
        ##############
        self.block2 = ResidualBlock(in_channels=64,out_channels=64,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        self.block2_b = ResidualBlock(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        ##############
        self.block3 = ResidualBlock(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        self.block3_b = ResidualBlock(in_channels=256,out_channels=256,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        ##############
        self.block4 = ResidualBlock(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        self.block4_b = ResidualBlock(in_channels=512,out_channels=512,kernel_size=(3,3),stride=2,padding=1,c7 = c7)
        ##############
        self.output_layer = nn.Linear(in_features= 512,out_features=10 )
    def forward(self,x):
        out1 = self.block1(self.input_layer(x))
        out1_b = self.block1_b(out1)
        #TODO
        ### need other block for the subgroups 
        out2 = self.block2(out1_b)
        out2_b = self.block2_b(out2)
        #TODO
        #####################
        out3 = self.block3(out2_b)
        out3_b = self.block3_b(out3)
        #TODO
        out4 = self.block4(out3_b)
        out4_b = self.block4_b(out4)
        #TODO
        #print(f"prior to linear layer: {out4_b.size()}")
        y = out4_b.view(out4_b.size(0),-1) ## flattening
        ### is this expected for outputlayer
        #print(f"output layer shape:{y.size()}, out4_b shape: {out4_b.size()}")
        ret = self.output_layer(y)#out4_b)
        return ret
##############################################
def q1(args,dataset):
    '''
    Run 2 epoch run 
    '''
    gpus_config = [[0],[0,1],[0,1,2,3]]
    batch_size = [32,128,512]
    '''
    vary gpu configs and batch_sizes
    '''
    sampler = get_sampler(dataset)
    dataloader = get_dataloader(sampler,args,b_size=batch)
    for gpu_id in gpus_config:
        for batch in batch_size:
            print(f"Current configuration batch size: {batch}, gpus: {len(gpu_id)}")
            model = create_model(args)
            model = DDP(model,gpu_id) ## gpu_id = [0],[0,1], or [0,1,2,3] 
            world_size = len(gpu_id) 
            setup(rank = 0,world_size = world_size)
            ############################### 
            cross_entropy = nn.CrossEntropyLoss()
            optimizer = optimizer_selection(model= model, opt = args.opt, lr = args.lr)
            ### 
            global epoch_time
            epoch_time= 0
            global mini_batch_time 
            mini_batch_time = 0
            global io_time
            io_time = 0
            ####### training loop##########
            for epoch in range(0,2):
                train(model,epoch,cross_entropy,optimizer,args.device,dataloader)
                if epoch ==0:
                    ### warmup
                    print("Warm-up epoch.....")
                    #epoch_time= 0
                    #mini_batch_time = 0
                    #io_time = 0
            print(f"Total times for epoch: {epoch_time} sec, mini batch computations: {mini_batch_time} sec, IO: {io_time} sec")
            print(f"Average Epoch time:{epoch_time/(5/5)}")### was orinally just foo/5
            print(f"Number of workers: {args.num_workers} sec")
            cleanup()

def train(model,epoch,criterion,optimizer,device,dataloader):
    print('\nEpoch: %d' % epoch)
    model.train()#resnet.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    mini_batch_times = []
    io_times = []
    torch.cuda.synchronize()## wait for kernels to finish....
    epoch_start = time.perf_counter()
    for batch_idx, (inputs, targets) in (enumerate(progress_bar)):#enumerate(trainloader):
        
        torch.cuda.synchronize()## wait for kernels to finish....
        io_start = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        torch.cuda.synchronize()## wait for kernels to finish....
        io_end = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()## wait for kernels to finish....torch.cuda.synchronize()## wait for kernels to finish....
        minibatch_end = time.perf_counter()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=100. * correct / total)
        mini_batch_times.append(minibatch_end-io_end)
        io_times.append(io_end-io_start)
        #print(f"\n minibatch :{minibatch_end-io_end}, io: {io_end-io_start}")
    torch.cuda.synchronize()## wait for kernels to finish....
    epoch_end = time.perf_counter()
    total_epoch = epoch_end-epoch_start
    print(f"epoch: {epoch} time:{total_epoch} sec")
    avg_mini_batch_time = torch.tensor(mini_batch_times).mean().item()
    avg_io_time = torch.tensor(io_times).mean().item()
    total_io = torch.tensor(io_times).sum().item()
    total_mini_batch = torch.tensor(mini_batch_times).sum().item()
    #######################################################
    average_loss = train_loss / len(dataloader)
    accuracy = correct / total
    print(f'Training Loss: {average_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')
    print(f"average mini batch time:{avg_mini_batch_time} sec, average I/O time: {avg_io_time} sec")
    print(f"mini batch time:{total_mini_batch} sec, I/O time: {total_io} sec\n")
    global epoch_time
    epoch_time+= total_epoch
    global mini_batch_time 
    mini_batch_time +=total_mini_batch 
    global io_time
    io_time += total_io 
    #return total_epoch,total_mini_batch,total_io

def optimizer_selection(model, opt,lr ):
    opt = opt.lower()
    print(f"opt: {opt} in the selection function")
    if opt == "sgd":
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif opt == "nesterov":
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4,nesterov=True)
    elif opt == "adadelta":
        ret = optim.Adadelta(model.parameters(), lr=lr,
                      weight_decay=5e-4)
    elif opt == 'adagrad':
        ret = optim.Adagrad(model.parameters(), lr=lr,
                    weight_decay=5e-4)
    elif opt == 'adam':
        ret = optim.Adam(model.parameters(), lr=lr,
                      weight_decay=5e-4)
    else:
        ### default sgd case:
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    return ret 

def create_model(args):
    device = args.device
    model = ResNet()
    model.to(device)
    return model 
### some DDP stuff ### 
## basically example code from the pytorch documentation 
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_sampler(dataset):
    return dist.DistributedSampler(dataset)
def get_dataloader(dataset,sampler,args,b_size):
    loader = torch.utils.data.DataLoader(
    dataset, batch_size=b_size, shuffle=False, num_workers=args.num_workers,sampler = sampler)
    return loader
if __name__ == "__main__":
    '''
    For loop and update the bach size
    '''

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda',type = str, help =  "device")
    parser.add_argument('--num_workers',default= 2, type= int, help = "dataloader workers")
    parser.add_argument('--data_path',default="./data", type= str, help = "data path")
    parser.add_argument('--opt', default ='sgd',type = str ,help = "optimzer")
    parser.add_argument('--c7', default=False,type= bool,help ="Question c7")
    args = parser.parse_args()
    #####################################
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ##################################
    trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
    q1(args,trainset)
    #train_sampler = dist.DistributedSampler(trainset) # new
    ##### TODO add sampler
    #trainloader_32 = torch.utils.data.DataLoader(
    #trainset, batch_size=32, shuffle=True, num_workers=args.num_workers,sampler = train_sampler) 
    ##################################
    #trainloader_128 = torch.utils.data.DataLoader(
    #trainset, batch_size=128, shuffle=True, num_workers=args.num_workers,sampler = train_sampler)
    ##################################
    #trainloader_512 = torch.utils.data.DataLoader(
    #trainset, batch_size=512, shuffle=True, num_workers=args.num_workers,sampler = train_sampler)
    ##################################
    #cross_entropy = nn.CrossEntropyLoss()
    #optimizer = optimizer_selection(model= model, opt = args.opt, lr = args.lr)

    '''
    ### loss same regardless
    global epoch_time
    epoch_time= 0
    global mini_batch_time 
    mini_batch_time = 0
    global io_time
    io_time = 0
    ############################################                                    
    for epoch in range(start_epoch, start_epoch+6):
        
        train(model,epoch,cross_entropy,optimizer,device,trainloader)
        if epoch == 0:
            print("Warm-up epoch.....")
            epoch_time= 0
            mini_batch_time = 0
            io_time = 0
            ## ignore epoch 0 
            #epoch_time+= dummy1
            #mini_batch_time+= dummy2
            #io_time+= dummy3
    print(f"Total times for epoch: {epoch_time} sec, mini batch computations: {mini_batch_time} sec, IO: {io_time} sec")
    print(f"Average Epoch time:{epoch_time/5}")
    print(f"Number of workers: {args.num_workers} sec")
    parameters_vs_gradients(model)
    '''