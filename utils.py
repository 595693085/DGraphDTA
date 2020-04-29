import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index = target_graph[tar_key]
            # print(np.array(features).shape, np.array(edge_index).shape)
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            # print(GCNData.target.size(), GCNData.target_edge_index.size(), GCNData.target_x.size())
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    TRAIN_BATCH_SIZE = 512
    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


#prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB
