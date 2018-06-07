from lenet import LeNet
from segmentation import segment
from houghtrans import hough

import torch
import fire
import sys
import cv2
import numpy as np
from PIL import Image
from pylab import *
from torchvision.transforms import ToTensor
#%matplotlib inline
font=cv2.FONT_HERSHEY_SIMPLEX

color =[(255,0,0),(139,0,0),(199,21,133),(255,69,0),(255,255,0),(123,104,238),(50,205,50),(128,128,0),(0,128,128),(0,0,205)]

def testresult(region):
	#输入一张图片region return这张图片上数字的预测值和它的置信度

	region = Image.fromarray(region)
	region = region.convert('1')
	region = ToTensor()(region)


	lenet = torch.load('./model/lenet.pth')

	lenet.cuda()
	lenet.eval()

	images=Variable(region.cuda()).cuda()
	outputs=lenet(images)
	predict=outputs.max(1)[1].data
	believe=1 #这里需要改成真正的置信度
	return [predict,believe]

def complete(inputAdd,outputAdd):
	for k in range(1):
			cv2.setUseOptimized(True)
			cv2.setNumThreads(4)
			im = cv2.imread(inputAdd)
			im = hough(im)
			newHeight = 1000
			newWidth = int(im.shape[1]*1000/im.shape[0])
			im = cv2.resize(im, (newWidth, newHeight))
			imOut=im.copy()
			box=segment(im)
			print (box)
			for i in range(0,len(box)):
				tmp = box[i]
				region=im[tmp[1]:tmp[3],tmp[0]:tmp[2]]
				#cv2.line(im,(tmp[0],tmp[1]),(tmp[2],tmp[3]),(255,255,0),2)
				#cv2.imwrite(str(k+1)+'_'+str(i)+'.jpg',region)
				#cv2.imwrite(str(k+1)+'______ .jpg', im)
				result=testresult(region)
				predict=result[0] #预测值
				believe=result[1] #置信度
				cv2.rectangle(imOut,(tmp[0],tmp[1]),(tmp[2],tmp[3]),color[predict], 1, cv2.LINE_AA)
				imOut=cv2.putText(imOut,str(believe),(tmp[0]-1,(tmp[1]+tmp[3])/2),font,1.2,color[predict],2)
				cv2.imwrite(outputAdd,imOut)

if __name__ == '__main__':
  fire.Fire(complete)
