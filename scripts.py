import os
import random
import numpy as np
import json
from collections import OrderedDict


def seq_format(proteins_dic, output_dir):
    # datasets = ['kiba', 'davis']
    # for dataset in datasets:
    # seq_path = os.path.join('data', dataset, 'seq_original')
    # fpath = os.path.join('data', dataset, 'proteins.txt')
    # proteins = json.load(fpath, object_pairs_hook=OrderedDict)
    for key, value in proteins_dic.items():
        with open(os.path.join(output_dir, key + '.fasta'), 'w') as f:
            f.writelines('>' + key + '\r\n')
            f.writelines(value + '\r\n')


def HHblitsMSA(bin_path, db_path, input_dir, output_dir):
    for fas_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, fas_file)
        output_file = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.hhr')  # igore
        output_file_a3m = os.path.join(output_dir, fas_file.split('.fasta')[0] + '.a3m')
        if os.path.exists(output_file) and os.path.exists(output_file_a3m):
            # print(output_file, output_file_a3m, 'exist.')
            # count += 1
            continue
        # print(process_file)
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        output_file_a3m = output_file_a3m.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' -maxfilt 100000 -realign_max 100000 -d ' + db_path + ' -all -B 100000 -Z 100000 -n 3 -e 0.001 -i ' + process_file + ' -o ' + output_file + ' -oa3m ' + output_file_a3m + ' -cpu 8'
        print(cmd)
        os.system(cmd)
        # print(cmd)
        # print(count)


def HHfilter(bin_path, input_dir, output_dir):
    file_prefix = []
    # print(input_dir)
    for file in os.listdir(input_dir):
        if 'a3m' not in file:
            continue
        temp_prefix = file.split('.a3m')[0]
        if temp_prefix not in file_prefix:
            file_prefix.append(temp_prefix)
    # random.shuffle(file_prefix)
    # print(len(file_prefix))
    # print(file_prefix)
    for msa_file_prefix in file_prefix:
        file_name = msa_file_prefix + '.a3m'
        process_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' -id 90 -i ' + process_file + ' -o ' + output_file
        print(cmd)
        os.system(cmd)


def reformat(bin_path, input_dir, output_dir):
    # print('reformat')
    for a3m_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, a3m_file)
        output_file = os.path.join(output_dir, a3m_file.split('.a3m')[0] + '.fas')
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = bin_path + ' ' + process_file + ' ' + output_file + ' -r'
        print(cmd)
        os.system(cmd)


def convertAlignment(bin_path, input_dir, output_dir):
    # print('convertAlignment')
    for fas_file in os.listdir(input_dir):
        process_file = input_dir + '/' + fas_file
        output_file = output_dir + '/' + fas_file.split('.fas')[0] + '.aln'
        if os.path.exists(output_file):
            continue
        process_file = process_file.replace('(', '\(').replace(')', '\)')
        output_file = output_file.replace('(', '\(').replace(')', '\)')
        cmd = 'python ' + bin_path + ' ' + process_file + ' fasta ' + output_file
        print(cmd)
        os.system(cmd)


def alnFilePrepare():
    import json
    from collections import OrderedDict
    print('aln file prepare ...')
    datasets = ['davis', 'kiba']
    # datasets = ['davis']
    for dataset in datasets:
        seq_dir = os.path.join('data', dataset, 'seq')
        msa_dir = os.path.join('data', dataset, 'msa')
        filter_dir = os.path.join('data', dataset, 'hhfilter')
        reformat_dir = os.path.join('data', dataset, 'reformat')
        aln_dir = os.path.join('data', dataset, 'aln')
        # pconsc4_dir = os.path.join('data', dataset, 'pconsc4')
        protein_path = os.path.join('data', dataset)
        proteins = json.load(open(os.path.join(protein_path, 'proteins.txt')), object_pairs_hook=OrderedDict)

        if not os.path.exists(seq_dir):
            os.makedirs(seq_dir)
        if not os.path.exists(msa_dir):
            os.makedirs(msa_dir)
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)
        if not os.path.exists(reformat_dir):
            os.makedirs(reformat_dir)
        if not os.path.exists(aln_dir):
            os.makedirs(aln_dir)

        HHblits_bin_path = '..../tool/hhsuite/bin/hhblits'  # HHblits bin path
        HHblits_db_path = '..../dataset/uniclust/uniclust30_2018_08/uniclust30_2018_08'  # hhblits dataset for msa
        HHfilter_bin_path = '..../tool/hhsuite/bin/hhfilter'  # HHfilter bin path
        reformat_bin_path = '..../tool/hhsuite/scripts/reformat.pl'  # reformat bin path
        convertAlignment_bin_path = '..../tool/CCMpred/scripts/convert_alignment.py'  # ccmpred convertAlignment bin path

        # check the programs used for the script
        if not os.path.exists(HHblits_bin_path):
            raise Exception('Program HHblits was not found. Please specify the run path.')

        if not os.path.exists(HHfilter_bin_path):
            raise Exception('Program HHfilter was not found. Please specify the run path.')

        if not os.path.exists(reformat_bin_path):
            raise Exception('Program reformat was not found. Please specify the run path.')

        if not os.path.exists(convertAlignment_bin_path):
            raise Exception('Program convertAlignment was not found. Please specify the run path.')

        seq_format(proteins, seq_dir)
        HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
        HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
        reformat(reformat_bin_path, filter_dir, reformat_dir)
        convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)

        print('aln file prepare over.')


def pconsc4Prediction():
    import pconsc4
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    # KTF.set_session(session)
    datasets = ['davis', 'kiba']
    model = pconsc4.get_pconsc4()
    for dataset in datasets:
        aln_dir = os.path.join('data', dataset, 'hhfilter')
        output_dir = os.path.join('data', dataset, 'pconsc4')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_list = os.listdir(aln_dir)
        random.shuffle(file_list)
        inputs = []
        outputs = []
        for file in file_list:
            input_file = os.path.join(aln_dir, file)
            output_file = os.path.join(output_dir, file.split('.a3m')[0] + '.npy')
            if os.path.exists(output_file):
                # print(output_file, 'exist.')
                continue
            inputs.append(input_file)
            outputs.append(output_file)
            try:
                print('process', input_file)
                pred = pconsc4.predict(model, input_file)
                np.save(output_file, pred['cmap'])
                print(output_file, 'over.')
            except:
                print(output_file, 'error.')



if __name__ == '__main__':
    alnFilePrepare()
    pconsc4Prediction()

