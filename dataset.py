import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils import data
from torchvision.transforms import ToTensor

class MNIST(data.Dataset):
	def __init__(self, root, subset):
		self.images_root = os.path.join(root, 'MNIST/')
		
		self.images_root += subset

		self.images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn]
		self.images.sort()

		self.labels = [f for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn]
		self.labels.sort()

	def __getitem__(self, index):
		image = self.load(self.images[index]).convert('L')
		image = ImageOps.invert(image)
		image = image.convert('1')
		image = ToTensor()(image)

		label = self.labels[index].split('_')[0]
		
		return image, int(label)

	def __len__(self):
		return len(self.images)

	def load(self, path): 
		return Image.open(path)
