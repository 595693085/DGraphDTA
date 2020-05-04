import sys, os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from collections import OrderedDict

from gnn import GNNNet
from utils import *
from emetrics import *
from data_process import *

pro_res_short_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y', 'X']
pro_res_long_table = ['Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln',
                      'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr', 'XXX']

pro_res_long_table_upper = [s.upper() for s in pro_res_long_table]
pro_res_map_dic = {p1.upper(): p2 for p1, p2 in zip(pro_res_long_table, pro_res_short_table)}


def target_to_graph_with_index(target_key, target_sequence, contact_dir, aln_dir, seq_index):
    target_edge_index = []
    target_size = len(seq_index)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    # print(target_feature.shape)
    # print(target_feature[seq_index, :].shape,len(seq_index))
    # print(seq_index)
    target_edge_index = np.array(target_edge_index)
    # print(target_feature[seq_index, :].shape,contact_map.shape)
    return target_size, target_feature[seq_index, :], target_edge_index


def create_dataset_for_comparison(pconsc4_contact_dir, real_contact_dir, test_csv, train_csv, prot_list):
    # pconsc4 can not predict this protein contact map because of memory limitation under our equipment
    df_train = pd.read_csv(train_csv)
    train_drugs_temp, train_prot_keys_temp, train_Y_temp = list(df_train['compound_iso_smiles']), list(
        df_train['target_key']), list(df_train['affinity'])
    df_test = pd.read_csv(test_csv)
    test_drugs_temp, test_prot_keys_temp, test_Y_temp = list(df_test['compound_iso_smiles']), list(
        df_test['target_key']), list(df_test['affinity'])

    train_drugs = []
    train_prot_keys = []
    train_Y = []
    test_drugs = []
    test_prot_keys = []
    test_Y = []

    # only select the protein with structures (proteins in prot_list)
    for i in range(len(train_prot_keys_temp)):
        if train_prot_keys_temp[i] in prot_list:
            train_drugs.append(train_drugs_temp[i])
            train_prot_keys.append(train_prot_keys_temp[i])
            train_Y.append(train_Y_temp[i])
    for i in range(len(test_prot_keys_temp)):
        if test_prot_keys_temp[i] in prot_list:
            test_drugs.append(test_drugs_temp[i])
            test_prot_keys.append(test_prot_keys_temp[i])
            test_Y.append(test_Y_temp[i])

    smile_graph = {}
    for smile in list(set(train_drugs + test_drugs)):
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    proteins = json.load(open('data/kiba/proteins.txt'), object_pairs_hook=OrderedDict)
    real_protein_graph = {}
    pconsc4_protein_graph = {}
    for prot in prot_list:
        # pconsc4: giving sequence and predicted contact map
        g = target_to_graph(prot, proteins[prot], pconsc4_contact_dir, aln_dir)
        pconsc4_protein_graph[prot] = g

        # real:cut sequence from sequence and real contact map
        real_seq_index = np.load(os.path.join(real_contact_dir, prot + '_index.npy'))

        # g_ = target_to_graph(key, proteins[key], real_protein_graph, aln_dir)
        # cut_seq = open(os.path.join(real_contact_dir, prot + '_seq.txt'), 'r').readline().strip()
        g_ = target_to_graph_with_index(prot, proteins[prot], real_contact_dir, aln_dir, real_seq_index)
        real_protein_graph[prot] = g_
        # print(g.shape, g_.shape)

    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)
    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)

    train_dataset_real = DTADataset(root='data', dataset='kiba', xd=train_drugs, target_key=train_prot_keys, y=train_Y,
                                    smile_graph=smile_graph, target_graph=real_protein_graph)
    train_dataset_pconsc4 = DTADataset(root='data', dataset='kiba', xd=train_drugs, target_key=train_prot_keys,
                                       y=train_Y, smile_graph=smile_graph, target_graph=pconsc4_protein_graph)

    test_dataset_real = DTADataset(root='data', dataset='kiba', xd=test_drugs, target_key=test_prot_keys, y=test_Y,
                                   smile_graph=smile_graph, target_graph=real_protein_graph)
    test_dataset_pconsc4 = DTADataset(root='data', dataset='kiba', xd=test_drugs, target_key=test_prot_keys,
                                      y=test_Y, smile_graph=smile_graph, target_graph=pconsc4_protein_graph)

    return train_dataset_real, test_dataset_real, train_dataset_pconsc4, test_dataset_pconsc4


def pdb_contat_map_parse_with_seq(seq, pdb_key, pdb_file, save_dir, cut_off=8):
    pdb_seq_from_pdb = {}
    with open(pdb_file, 'r') as f:
        # print('test open')
        chain_id_last = ''
        for line in f.readlines():
            # print('test read')
            if line[:6] == 'SEQRES':
                # print('test SEQRES')
                # example: SEQRES  17 B  286  VAL CYS LYS PRO GLU GLU ARG PHE ARG ALA PRO PRO ILE
                line_arr = line.split()
                chain_id_temp = line_arr[2]
                if chain_id_temp != chain_id_last:
                    # print('test1')
                    chain_id_last = chain_id_temp
                    pdb_seq_from_pdb[chain_id_temp] = ''
                # seq_temp = [pro_res_map_dic[s] for s in line_arr[4:]] # raise error
                seq_temp = []
                for s in line_arr[4:]:
                    if s in pro_res_map_dic.keys():
                        seq_temp.append(pro_res_map_dic[s])
                    else:
                        # print(s)
                        seq_temp.append('X')
                pdb_seq_from_pdb[chain_id_temp] += ''.join(seq_temp)
        # print(pdb_seq_from_pdb)
        # print(seq)  #for check
        chain_id_exist = ''
        for chain_id in pdb_seq_from_pdb.keys():
            # if seq in pdb_seq_from_pdb[chain_id]: raise error for few residues
            # because there may be several different residues between the giving seq and the pdb seq
            # so only the beginning 20 resiudes and the last 20 residues are used to locate the seq
            if seq[:20] in pdb_seq_from_pdb[chain_id] and seq[-20:] in pdb_seq_from_pdb[chain_id]:
                chain_id_exist = chain_id
                break
        assert chain_id_exist != ''

        seq_pdb = pdb_seq_from_pdb[chain_id_exist]
        # the giving seq is corresponding to seq_pdb[mark_start:mark_start + len(seq)]
        mark_start = seq_pdb.find(seq[:20])
        # second check for determintation of this piece are coresponding to the giving seq
        assert mark_start != -1 and seq[-20:] == seq_pdb[mark_start:mark_start + len(seq)][-20:]
        # res_coor = np.zeros((len(seq), 3))  # for saving coordinates
        res_coor = []
    with open(pdb_file, 'r') as f:
        # count = 0
        # line_last = ''
        seq_list = []
        seq_index_list = []  # for saving the indexes for those residues have coordinates
        for line in f.readlines():
            # print(line)
            if line[:4] == 'ATOM':
                # example: ATOM    283  NZ  LYS A  45     -16.618   0.496 199.470  1.00 54.83           N
                if line[21] != chain_id_exist:
                    continue
                res_name = pro_res_map_dic[line[17:20]]
                seq_index = int(line[22:26])
                if seq_index < mark_start and seq_index >= mark_start + len(seq):  # not in the targeted seq
                    continue
                if line[12:16].strip() == 'CB' or (
                        res_name == 'G' and line[12:16].strip() == 'CA'):  # C beta coordinate

                    # count += 1
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    # print(pdb_file)
                    # print(line)
                    # print(seq_index, [x, y, z])
                    # res_coor[seq_index - 1] = np.array()

                    if (seq_index - mark_start - 1) in seq_index_list:  # AVAL and BVAL, need process
                        continue
                    res_coor.append([x, y, z])
                    seq_index_list.append(seq_index - mark_start - 1)
                    seq_list.append(res_name)
                    # line_last = line

        # print(len(list(set(seq_index_list))), len(seq))
        # print(seq_index_list)
        # print(pdb_file)
        assert len(list(set(seq_index_list))) == len(seq_index_list)  # ensure no repeated index
        res_coor = np.array(res_coor)
        # for i in range(res_coor.shape[0]):
        #     print(res_coor[i])
        # print(res_coor.shape)
        seq_piece_len = len(seq_index_list)
        dis_np = np.zeros((seq_piece_len, seq_piece_len))
        # contact map
        for i in range(seq_piece_len):
            for j in range(seq_piece_len):
                dis_np[i][j] = np.linalg.norm(res_coor[i] - res_coor[j])
                # print(pdb_file,res_coor[i], res_coor[j], dis_np[i][j])
        contact_map = np.where(dis_np < cut_off, 1.0, 0.0)
        # print(contact_map,type(contact_map))
        contact_map -= np.matrix(np.eye(contact_map.shape[0]))

        # not all residues have their coordinates, there are some missing residues,
        # so the contact map is a piece of the whole seq,  also save the indexes for residues and contact map
        np.save(os.path.join(save_dir, pdb_key + '.npy'), contact_map)
        np.save(os.path.join(save_dir, pdb_key + '_index.npy'), np.array(seq_index_list))
        open(os.path.join(save_dir, pdb_key + '_seq.txt'), 'w').writelines(''.join(seq_list))
        # print(''.join(seq_list))

        # return contact_map, seq_index_list


def contact_map_compare(real_contact_map, pconsc4_contact_map):
    # print(real_contact_map.shape, pconsc4_contact_map.shape)
    difference_map = (real_contact_map - pconsc4_contact_map).reshape(-1)
    accurate_count = len(np.where(difference_map == 0.0)[0])
    return accurate_count / len(difference_map)


def validation_experiment(model, device, test_csv, real_contact_dir, pconsc4_contact_dir, aln_dir, prot_list):
    df_test = pd.read_csv(test_csv)
    test_drugs_temp, test_prot_keys_temp, test_Y_temp = list(df_test['compound_iso_smiles']), list(
        df_test['target_key']), list(
        df_test['affinity'])

    # check whether the protein has a whole structure,
    test_drugs = []
    test_prot_keys = []
    test_Y = []
    for i in range(len(test_drugs_temp)):
        if test_prot_keys_temp[i] in prot_list:
            test_drugs.append(test_drugs_temp[i])
            test_prot_keys.append(test_prot_keys_temp[i])
            test_Y.append(test_Y_temp[i])

    # print('test_Y:',len(test_Y))

    smile_graph = {}
    for smile in list(set(test_drugs)):
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    proteins = json.load(open('data/kiba/proteins.txt'), object_pairs_hook=OrderedDict)
    real_protein_graph = {}
    pconsc4_protein_graph = {}
    for key in prot_list:
        cut_seq = open(os.path.join(real_contact_dir, key + '_seq.txt'), 'r').readline().strip()
        # g = target_to_graph(key, proteins[key], real_contact_dir, aln_dir)
        g = target_to_graph(key, cut_seq, real_contact_dir, aln_dir)
        real_protein_graph[key] = g
        # g_ = target_to_graph(key, proteins[key], pconsc4_contact_dir, aln_dir)
        g_ = target_to_graph(key, cut_seq, pconsc4_contact_dir, aln_dir)
        pconsc4_protein_graph[key] = g_
        # print(g.shape, g_.shape)

    TEST_BATCH_SIZE = 32
    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(
        test_Y)
    test_dataset_real = DTADataset(root='data', dataset='kiba_test', xd=test_drugs, target_key=test_prot_keys, y=test_Y,
                                   smile_graph=smile_graph, target_graph=real_protein_graph)
    test_dataset_pconsc4 = DTADataset(root='data', dataset='kiba_test', xd=test_drugs, target_key=test_prot_keys,
                                      y=test_Y, smile_graph=smile_graph, target_graph=pconsc4_protein_graph)
    test_loader_real = torch.utils.data.DataLoader(test_dataset_real, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                   collate_fn=collate)
    test_loader_pconsc4 = torch.utils.data.DataLoader(test_dataset_pconsc4, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                      collate_fn=collate)

    print(len(test_dataset_real), len(test_dataset_pconsc4))
    G1, P_real = predicting(model, device, test_loader_real)
    G2, P_pconsc4 = predicting(model, device, test_loader_pconsc4)

    assert (G1 == G2).all()
    mse_real = get_mse(G1, P_real)
    ci_real = get_cindex(G1, P_real)
    perason_real = get_pearson(G1, P_real)

    mse_pconsc4 = get_mse(G2, P_pconsc4)
    ci_pconsc4 = get_cindex(G2, P_pconsc4)
    perason_pconsc4 = get_pearson(G2, P_pconsc4)

    print('real:', mse_real, ci_real, perason_real)
    print('pconsc4:', mse_pconsc4, ci_pconsc4, perason_pconsc4)


if __name__ == '__main__':
    # seq = 'MGNTSSERAALERHGGHKTPRRDSSGGTKDGDRPKILMDSPEDADLFHSEEIKAPEKEEFLAWQHDLEVNDKAPAQARPTVFRWTGGGKEVYLSGSFNNWSKLPLTRSHNNFVAILDLPEGEHQYKFFVDGQWTHDPSEPIVTSQLGTVNNIIQVKKTDFEVFDALMVDSQKCSDVSELSSSPPGPYHQEPYVCKPEERFRAPPILPPHLLQVILNKDTGISCDPALLPEPNHVMLNHLYALSIKDGVMVLSATHRYKKKYVTTLLYKPI'
    # pdb_contat_map_parse_with_seq(seq, 'test', '4CFF.pdb', '/', cut_off=8)

    cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[1])]
    flag = ['real', 'pconsc4'][int(sys.argv[2])]
    dataset = 'kiba'
    print('cuda_name:', cuda_name)
    print('which experiment:', flag)

    # directories
    prot_list = os.listdir('data/validation_data/structures')  # select pdb structures manually
    proteins = json.load(open('data/kiba/proteins.txt'), object_pairs_hook=OrderedDict)
    real_contact_dir = 'data/validation_data/real_contact'
    pconsc4_contact_dir = 'data/kiba/pconsc4'
    aln_dir = 'data/kiba/aln'
    train_csv = 'data/kiba_train.csv' # can be obtained after running training.py
    test_csv = 'data/kiba_test.csv'  # can be obtained after running training.py

    if not os.path.exists(real_contact_dir):
        os.makedirs(real_contact_dir)

    # generate contact map according to the giving sequence
    for prot in prot_list:
        if os.path.exists(os.path.join(real_contact_dir, prot + '.npy')):
            continue
        seq = proteins[prot]  # giving sequence in kiba dataset
        pdb_files = os.listdir(os.path.join('data/validation_data/structures', prot))  # the corresponding structures
        for i in range(len(pdb_files)):
            pdb_file = os.path.join('data/validation_data/structures', prot, pdb_files[i])
            try:
                pdb_contat_map_parse_with_seq(seq, prot, pdb_file, real_contact_dir, cut_off=8)  # parse the contact map
                break
            except:
                print('can not create contact map from', pdb_file, 'for', prot)
                # import traceback
                #
                # traceback.print_exc()
                # exit(-1)
                continue
    # validation contact map accuracy
    prot_valid_list = []
    for prot in prot_list:
        if os.path.exists(os.path.join(real_contact_dir, prot + '_seq.txt')):
            prot_valid_list.append(prot)
    # pconsc4 can not predict this protein contact map because of memory limitation under our equipment
    if 'P78527' in prot_valid_list:
        prot_valid_list.remove('P78527')

    # contact map accuracy count
    accuracys = []
    for prot in prot_valid_list:
        # print(len(proteins[prot]))
        seq_index = np.load(os.path.join(real_contact_dir, prot + '_index.npy'))
        real_contact_map = np.load(os.path.join(real_contact_dir, prot + '.npy'))
        pconsc4_contact_map = np.load(os.path.join(pconsc4_contact_dir, prot + '.npy'))
        pconsc4_contact_map_cut = pconsc4_contact_map[seq_index, :][:, seq_index]
        pconsc4_contact_map_cut = np.where(pconsc4_contact_map_cut >= 0.5, 1.0, 0.0)
        # print(pconsc4_contact_map_cut.shape, real_contact_map.shape)
        # for i in range(real_contact_map.shape[0]):
        #     for j in range(real_contact_map.shape[1]):
        #         print(i, j, real_contact_map[i, j], pconsc4_contact_map_cut[i, j])
        acuracy = contact_map_compare(real_contact_map, pconsc4_contact_map_cut)
        accuracys.append(acuracy)

    print('accuracy of predicted contact map:', np.mean(np.array(accuracys)))

    # create dataset
    TRAIN_BATCH_SIZE = 512
    Test_BATCH_SIZE = 512
    NUM_EPOCHS = 0
    LR = 0.001
    dataset_train_real, dataset_test_real, dataset_train_pconsc4, dataset_test_pconsc4 = create_dataset_for_comparison(
        pconsc4_contact_dir, real_contact_dir, test_csv, train_csv, prot_valid_list)
    print('train_size:', len(dataset_train_real), len(dataset_train_pconsc4))
    print('test_size:', len(dataset_test_real), len(dataset_test_pconsc4))
    dataloader_train_real = torch.utils.data.DataLoader(dataset_train_real, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                        collate_fn=collate)
    dataloader_test_real = torch.utils.data.DataLoader(dataset_test_real, batch_size=Test_BATCH_SIZE, shuffle=False,
                                                       collate_fn=collate)
    dataloader_train_pconsc4 = torch.utils.data.DataLoader(dataset_train_pconsc4, batch_size=TRAIN_BATCH_SIZE,
                                                           shuffle=True, collate_fn=collate)
    dataloader_test_pconsc4 = torch.utils.data.DataLoader(dataset_test_pconsc4, batch_size=Test_BATCH_SIZE,
                                                          shuffle=False, collate_fn=collate)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    # model_file_name = 'models/model_' + model_st + '_' + dataset + '.model'
    model = GNNNet()
    model.to(device)
    model_st = GNNNet.__name__
    model_file_name = 'models/model_' + model_st + '_' + flag + '.model'
    result_file_name = 'results/result_' + model_st + '_' + flag + '.txt'
    model.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    result_str = ''
    if flag == 'real':
        print('real check')
        # exit(-1)
        for epoch in range(NUM_EPOCHS):
            train(model, device, dataloader_train_real, optimizer, epoch + 1)
            G, P = predicting(model, device, dataloader_test_real)
            val = get_mse(G, P)
            # print('test mse:', val, best_mse, 'test_loader')
            if val < best_mse:
                best_mse = val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)
                print('Mse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, flag)
            else:
                print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, flag)
        G, P = predicting(model, device, dataloader_test_real)
        ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_ci(G, P), get_rm2(G, P)]
        result_str += flag + '\r\n'
        result_str += 'rmse:' + str(ret[0]) + ' mse:' + str(ret[1]) + ' pearson:' + str(ret[2]) + 'spearman:' + str(
            ret[3]) + 'ci:' + str(ret[4]) + 'rm2:' + str(ret[5])
        open(result_file_name, 'w').writelines(result_str)
    else:
        print('pconsc4 check')
        # exit(-1)
        for epoch in range(NUM_EPOCHS):
            train(model, device, dataloader_train_pconsc4, optimizer, epoch + 1)
            G, P = predicting(model, device, dataloader_test_pconsc4)
            val = get_mse(G, P)
            # print('test mse:', val, best_mse, 'test_loader')
            if val < best_mse:
                best_mse = val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)
                print('Mse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, flag)
            else:
                print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, flag)

        G, P = predicting(model, device, dataloader_test_pconsc4)
        ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_ci(G, P), get_rm2(G, P)]
        result_str += flag + '\r\n'
        result_str += 'rmse:' + str(ret[0]) + ' mse:' + str(ret[1]) + ' pearson:' + str(ret[2]) + 'spearman:' + str(
            ret[3]) + 'ci:' + str(ret[4]) + 'rm2:' + str(ret[5])
        open(result_file_name, 'w').writelines(result_str)
