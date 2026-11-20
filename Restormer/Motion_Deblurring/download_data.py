import os
import shutil
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='GoPro', help='all, GoPro, HIDE, RealBlur_R, RealBlur_J')
args = parser.parse_args()
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'
GoPro_test = '1k6DTSHu4saUgrGTYkkZXTptILyG9RRll'
HIDE_test = '1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A'
RealBlurR_test = '1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS'
RealBlurJ_test = '1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW'
dataset = args.dataset
for data in args.data.split('-'):
    if data == 'train':
        print('GoPro Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
        os.system(f'gdrive download {GoPro_train} --path Datasets/Downloads/')
        print('Extracting GoPro data...')
        shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
        os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'GoPro'))
        os.remove('Datasets/Downloads/train.zip')
    if data == 'test':
        if dataset == 'all' or dataset == 'GoPro':
            print('GoPro Testing Data!')
            os.system(f'gdrive download {GoPro_test} --path Datasets/')
            print('Extracting GoPro Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')
        if dataset == 'all' or dataset == 'HIDE':
            print('HIDE Testing Data!')
            os.system(f'gdrive download {HIDE_test} --path Datasets/')
            print('Extracting HIDE Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')
        if dataset == 'all' or dataset == 'RealBlur_R':
            print('RealBlur_R Testing Data!')
            os.system(f'gdrive download {RealBlurR_test} --path Datasets/')
            print('Extracting RealBlur_R Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')
        if dataset == 'all' or dataset == 'RealBlur_J':
            print('RealBlur_J testing Data!')
            os.system(f'gdrive download {RealBlurJ_test} --path Datasets/')
            print('Extracting RealBlur_J Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')
