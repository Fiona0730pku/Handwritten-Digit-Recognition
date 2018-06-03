from lenet import LeNet
from dataset import MNIST

import PIL
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable

def test():
	test_dataset = MNIST(root='.', subset='test')
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	lenet = torch.load('./model/lenet.pth')

	lenet.cuda()
	lenet.eval()

	s = 0
	n = 0
	for i, (images, labels) in enumerate(test_loader):
		images = Variable(images.cuda()).cuda()
		labels = Variable(labels.cuda()).cuda()

		outputs = lenet(images)

		s = s + (outputs.max(1)[1].data == labels.data).sum()
		n = n + 1

	print(s * 1.0 / n)



def train():
	train_dataset = MNIST(root='.', subset='train')
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

	lenet = LeNet()

	optimizer = Adam(lenet.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss()

	lenet.cuda()
	lenet.train()

	for epoch in range(10):
		print("Epoch {} : ".format(epoch + 1))
		if epoch == 5:
			optimizer = Adam(lenet.parameters(), lr=0.001)

		for i, (images, labels) in enumerate(train_loader):
			images = Variable(images.cuda(), requires_grad=True).cuda()
			labels = Variable(labels.cuda()).cuda()

			outputs = lenet(images)

			optimizer.zero_grad()
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

		torch.save(lenet, './model/lenet.pth')

		test()

train()