import sys
sys.path.append('./')
import argparse
import pdb
import argparse
import yaml 
from torch.utils.data import DataLoader
from trainer import ModelTrainer
from tester import ModelTester
from utils.setup import setup_solver
import os
import pickle
# from model import MYNET
from model_std2 import MYNET
# from model_no_att import MYNET
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self,dataset='RLVD',trainset=True):
        self.trainset=trainset
        if trainset:
            self.data,self.labels = np.load(os.path.join('datasets',dataset,'train','data.npy')),\
                                    np.load(os.path.join('datasets',dataset,'train','labels.npy'))
        else:
            self.data,self.labels = np.load(os.path.join('datasets',dataset,'test','data.npy')),\
                                    np.load(os.path.join('datasets',dataset,'test','labels.npy'))

    def __len__(self):
        return len(self.data[:,0,0,0])

    def get_data(self):
        return self.data,self.labels

    def __getitem__(self, index):
        return self.data[index],self.labels[index]


def train(args):
    with open(os.path.join(args.config,args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    model = MYNET(**config['MYNET'])
    
    # if args.dataset == 'RWF':
    # from datasets.RWF import RWF
    train_dataset =  dataset(dataset='RLVD',trainset=True)
    valid_dataset = dataset(dataset='RLVD', trainset=False)
    # train_dataset = RWF(config['datasets']['train'], 'train', config['MYNET']['sequence_size'], temporal_stride=config['datasets']['stride'])


    # valid_dataset = RWF(config['datasets']['valid'], 'valid', config['MYNET']['sequence_size'])
    
    train_loader = DataLoader(train_dataset, **config['dataloader']['train'])
    valid_loader = DataLoader(valid_dataset, **config['dataloader']['valid'])

    optimizer, scheduler, criterion = setup_solver(model.parameters(), config)
    config['trainer']['device'] = 0
    trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, config, config['criterion']['name'], dataset_name=config['datasets']['name'], **config['trainer'])
    trainer.train()


def test(args):
    with open(os.path.join(args.config,args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    model = MYNET(**config['MYNET'])
    
    if args.dataset == 'RWF':
        from datasets.RWF_test import RWF_test
        test_dataset = RWF_test(config['datasets']['test'], config['MYNET']['sequence_size'])

    test_loader = DataLoader(test_dataset, **config['dataloader']['test'])
    tester = ModelTester(model, test_loader, **config['tester'])
    output = tester.test()
    with open("output.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    args = parser.parse_args()

    seed = range(1501,2000)

    for seed_num in seed:
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)

        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed_num)

        print("seed: {}".format(str(seed_num)))
        if args.mode == 'Train':
            train(args)
        elif args.mode == 'Test':
            test(args)
