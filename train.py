# @ Author: Wenbo Duan
# @ Email: bobbyduanwenbo@live.com
# @ Date: 23, Sep 2023
# @ Function: main script of training the model


import torch
from torch.utils.tensorboard import SummaryWriter
from customize_set import *
from model_zoo import *
from tqdm import tqdm
import wandb
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.utils import add_self_loops
import copy
from customize_encoder import *
from torch.optim.lr_scheduler import StepLR

##### TRAINING CONFIG #####
positional_encoding = True

if positional_encoding:
    PE_feature_num = 30
    transform_globalPE=T.AddLaplacianEigenvectorPE(k=10,is_undirected=True,attr_name='pos_globalPE')
    transform_r=T.AddRandomWalkPE(walk_length=10, attr_name='pos_r')
    transform_localPE = AddLocalPE(walk_length=10, attr_name='pos_localPE')
    transform_localSE = AddLocalSE(walk_length=10, attr_name='pos_localSE')
    transform_degree = AddDegree('degree')

dataset = GC_20_aug('./data/GC_20_aug/')

lr = 3e-4
epoch = 200
batch_size = 128
single_shot = True
train_num = 1 if single_shot else 8900
test_num = 1
use_wandb = False
project_name = 'thesis_NC'
device = 'cuda'
min_node_size = 5
hidden = 128
num_layer = 3

model_backbone = GPSS(hidden, PE_feature_num, num_layer).to(device)
# model_backbone = GINE(hidden, PE_feature_num, num_layer).to(device)
optimizer_backbone = torch.optim.Adam(model_backbone.parameters(), lr=lr, weight_decay=5e-4)
criterion_backbone = torch.nn.BCELoss()
# scheduler_backbone = CosineAnnealingWarmRestarts(optimizer_backbone, T_0=2500)
scheduler = StepLR(optimizer_backbone, step_size=50, gamma=0.5)

model_head = torch.nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()).to(device) 
optimizer_head = torch.optim.Adam(model_head.parameters(), lr=lr, weight_decay=5e-4)
criterion_head = nn.BCELoss()

checkpoint_backbone = 'NCC_20_to50.pth'
checkpoint_head = 'head.pth'    

#########################

def load_backbone(checkpoint_path):
    print(f'loadining {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    # model = NC_1(dataset.num_features, hidden=128).to(device)
    
    # model = GPSS(128, 8, 3).to(device) *_002.pth
    model = GPSS(128, 30, 3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    last_loss = checkpoint['loss']
    print(f'last loss:{last_loss}\n')
    return model





def add_self_loop(data):
    edge_index = add_self_loops(data.edge_index, data.edge_attr)
    new_data = copy.copy(data)
    new_data.edge_index = edge_index[0]
    new_data.edge_attr = edge_index[1]
    return new_data


def encode(data):
    data = transform_degree(data)
    data = add_self_loop(data)
    data = transform_localPE(data)

    # data = transform_globalPE(data)
    # data.pos_globalPE = data.pos_globalPE.to(device)
    # data = transform_localSE(data)
    data = transform_r(data)


    return data



def parallel(model):
    if torch.cuda.device_count() > 1:
        return nn.DataParallel(model).to(device)
    else:
        return model
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)


def build_data(raw_data):
    train_set = DataLoader(raw_data, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(raw_data[train_num:train_num+test_num], batch_size=batch_size, shuffle=True)
    return train_set, test_set


def train(data):
    model_backbone.train()
    optimizer_backbone.zero_grad()
    data_loop = encode(data)
    y_pred = model_backbone(data_loop.degree, data_loop.pos_localPE,data.edge_index,  data.batch, data.edge_attr)
    loss = criterion_backbone(y_pred, data.y)

    loss.backward()
    optimizer_backbone.step()
    return loss.item()


def test(test_set):
    model_backbone.eval()
    cu_loss = 0
    for data in test_set:
        data = data.to(device)
        data_loop = encode(data)
        y_pred = model_backbone(data_loop.degree,data_loop.pos_localPE, data.edge_index, data.batch, data.edge_attr)
        loss = criterion_backbone(y_pred, data.y)

        cu_loss += loss.item()
    return cu_loss


def openwandb():  
    wandb.init(project=project_name,
               sync_tensorboard = True,
               config={
                   "learing rate": lr,
                   "epochs":epoch,
                   "traing set size": len(train_set),
                   "node size":20,
                   "model name": checkpoint_backbone
               })
    wandb.watch(model_backbone, log_freq=100)
    wandb.Artifact(
    name= project_name,
    type='model'
    )

def add_checkpoint(model, optimizer, path, epoch, loss):
    torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss
                },
                f'{path}')


def load_backbone(checkpoint_path):
    print(f'loadining {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model = NC_1(dataset.num_features, hidden=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    last_loss = checkpoint['loss']
    print(f'last loss:{last_loss}\n')
    return model



def onehot(sample):
    node_size = sample.x.shape[0]
    one_hot = torch.zeros(node_size,1)
    one_hot[int(sample.y)] = 1
    return one_hot
    


train_set, test_set = build_data(dataset)
print(f'Batch Size: {batch_size}\nInput shape: {list(test_set)[0].x.shape}\nOutput shape: {list(test_set)[0].y.shape}')

# Training the embedding model


###### Main Loop #######
## online recording
if use_wandb:
    openwandb()

history = []
## Recording the computation graph
dummy_sample = list(train_set)[0].to(device)
writer = SummaryWriter()
model_backbone.eval()

batch_num_train = len(train_set)
batch_num_test = len(test_set)
iters = len(train_set)
for i in (bar := tqdm(range(epoch))):
    train_loss = 0
    test_loss = 0
    for j, data in enumerate(train_set):
        data = data.to(device)
        loss = train(data)
        train_loss = train_loss +  loss
        # scheduler_backbone.step(i+j/iters)
    train_loss = train_loss / batch_num_train

    # scheduler.step()

    if not single_shot:
        test_loss = test(test_set)
        test_loss = test_loss / batch_num_test

    history.append(train_loss)
    bar.set_description(f'EPOCH: {i+1}/{epoch} |TRAINING LOSS: {train_loss:.10f} | TEST LOSS: {test_loss}')
    writer.add_scalar('Loss/train', train_loss, i)
    writer.add_scalar('Loss/Test', test_loss, i)

    # add check point
    add_checkpoint(model_backbone,optimizer_backbone,checkpoint_backbone,i,train_loss)
    
# visualize(data)
print(f'INPUT SIZE :{dataset[0].x.shape}')
print(f'INPUT PREVIEW :{dataset[0].x[:10].T}')    