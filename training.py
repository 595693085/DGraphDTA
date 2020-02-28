import sys, os
import torch
import torch.nn as nn
from gnn import GNNNet

from utils import *
from emetrics import *
from create_data import create_dataset

datasets = [['davis', 'kiba'][int(sys.argv[1])]]
model_st = GNNNet.__name__

cuda_name = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"][int(sys.argv[2])]
print('cuda_name:', cuda_name)

fold = [0, 1, 2, 3, 4][int(sys.argv[3])]
# print(int(sys.argv[3]))
print("fold", fold)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
LOG_INTERVAL = 10
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = "models"
results_dir = "results"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
# datasets=[]
for dataset in datasets:
    train_data, valid_data, test_data = create_dataset(dataset, fold)
    # train_size = int(0.8 * len(train_data))
    # valid_size = len(train_data) - train_size
    # print("length", train_size, valid_size)
    # train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    # make data PyTorch mini-batch processing ready

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                              collate_fn=collate)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = GNNNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_test_rmse = 1000
    best_test_mse = 1000
    best_test_ci = 0
    best_epoch = -1
    model_file_name = 'models/model_' + model_st + '_' + dataset + "_" + str(fold) + '.model'
    result_file_name = 'results/result_' + model_st + '_' + dataset + "_" + str(fold) + '.txt'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val = get_mse(G, P)
        print("val mse:", val, best_mse, "valid_loader")
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model, model_file_name)
            print('predicting for test data')
            G, P = predicting(model, device, test_loader)
            # ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]

            # calculation of the 4 results is very slow
            ret = get_mse(G, P)
            # with open(result_file_name, 'w') as f:
            #     f.write(','.join(map(str, ret)))
            # best_test_rmse = ret[0]
            best_test_mse = ret
            # best_test_ci = ret[2]
            print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_test_mse, model_st, dataset, fold)
        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse', best_test_mse, model_st, dataset, fold)

    # for results save
    model = torch.load(model_file_name)
    G, P = predicting(model, device, test_loader)
    ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_ci(G, P)]
    result_str += dataset + 'fold' + str(fold) + '\r\n'
    result_str += 'rmse:' + str(ret[0]) + ' mse:' + str(ret[1]) + ' pearson:' + str(ret[2]) + 'spearman:' + str(
        ret[3]) + 'ci:' + str(ret[4])
    open(result_file_name, 'w').writelines(result_str)
