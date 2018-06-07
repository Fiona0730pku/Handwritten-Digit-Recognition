from lenet import LeNet
from segmentation import segment
from hough import hough

import torch
import fire
import sys
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pylab import *
from torch.autograd import Variable
from torchvision.transforms import ToTensor
#%matplotlib inline
font=cv2.FONT_HERSHEY_SIMPLEX

color =[(255,0,0),(139,0,0),(199,21,133),(255,69,0),(255,255,0),(123,104,238),(50,205,50),(128,128,0),(0,128,128),(0,0,205)]

def testresult(region):
	#输入一张图片region return这张图片上数字的预测值和它的置信度
	region = Image.fromarray(region)
	w, h = region.size
	if w >= h:
		region = region.resize((28, 28))
	else:
		newWidth = int(w * 28 / h)
		newImage = Image.new("RGB", (28, 28), color=(255, 255, 255))
		region = region.resize((newWidth, 28))
		p = int((28 - newWidth) / 2)
		newImage.paste(region, (p, 0))
		region = newImage
	region = region.filter(ImageFilter.MinFilter)
	newImage = Image.new("RGB", (32, 32), color=(255, 255, 255))
	newImage.paste(region, (2, 2))
	region = newImage
	region = region.convert('1')
	region = region.resize((28, 28))
	region = ToTensor()(region)


	lenet = torch.load('C:/Users/李思航/Documents/GitHub/Handwritten-Digit-Recognition/model/lenet.pth')

	lenet.cuda()
	lenet.eval()

	images=Variable(region.cuda()).cuda()
	images = images.view(1,1,28,28)
	outputs=lenet(images)
	outputs = F.softmax(outputs)
	predict=outputs.max(1)[1].data[0]
	believe=outputs.data[0][predict] #这里需要改成真正的置信度
	believe=round(believe, 2)
	print(predict, believe)
	return predict, believe

def complete(inputAdd,outputAdd):
	im = cv2.imread(inputAdd)
	imOut=im.copy()
	im = hough(im)
	box=segment(im)
	print (box)
	for i in range(0,len(box)):
		tmp = box[i]
		region=im[tmp[1]:tmp[3],tmp[0]:tmp[2]]
		predict, believe = testresult(region)
		cv2.rectangle(imOut,(tmp[0],tmp[1]),(tmp[2],tmp[3]),color[predict], 1, cv2.LINE_AA)
		imOut=cv2.putText(imOut,str(predict)+" "+str(believe),(tmp[0]-20,int((tmp[1]+tmp[3])/2)),font,1,color[predict],2)
	cv2.imwrite(outputAdd,imOut)

if __name__ == '__main__':
	fire.Fire(complete)
