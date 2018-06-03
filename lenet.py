import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 6, 3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(6, 16, 5, stride=1, padding=0),
			nn.MaxPool2d(2, 2))

		self.fc = nn.Sequential(
			nn.Linear(400, 120),
			nn.Linear(120, 84),
			nn.Linear(84, 10))

	def forward(self, x):
		out = self.conv(x)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out