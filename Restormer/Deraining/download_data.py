import os
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
args = parser.parse_args()
rain13k_train = '14BidJeG4nSNuFNFDf99K-7eErCq4i47t'
rain13k_test = '1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8'
for data in args.data.split('-'):
    if data == 'train':
        print('Rain13K Training Data!')
        os.system(f'gdrive download {rain13k_train} --path Datasets/')
        print('Extracting Rain13K data...')
        shutil.unpack_archive('Datasets/train.zip', 'Datasets')
        os.remove('Datasets/train.zip')
    if data == 'test':
        print('Download Deraining Testing Data')
        os.system(f'gdrive download {rain13k_test} --path Datasets/')
        print('Extracting test data...')
        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
        os.remove('Datasets/test.zip')
