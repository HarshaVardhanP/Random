#python belfer_pyt.py /home/harsha/Desktop/Datasets/DumData

from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch Benchmark')
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')


def main():
	t = time.time()
	global args
    	args = parser.parse_args()
	
	valdir = os.path.join(args.data,'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
	val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
	batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

	print(valdir)

	#loading pretrained model
	net = models.alexnet(pretrained=True)
	net.cuda()
	net.eval()
	print(net)

	#passing images thru network
	n = 1
        batch_avgtime=0
	for data, target in val_loader:
		td = time.time()
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		p = net(data)
                #print(p)
                batch_time = time.time()-td	
                batch_avgtime=batch_avgtime+batch_time
		print('Batch:',n,'Time:',batch_time)
                if n==20:
                    break
                n=n+1
				
	tf = time.time()-t
        #print('Avg Batch Time:',batch_avgtime/n)
        #print('Total Time:',tf)
	#print('Total Batches:',n)
	
if __name__ == '__main__':
    main()




