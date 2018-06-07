from segmentation import segment

import torch
import sys
import cv2
import numpy as np

def test(region):

	

for k in range(1):
		cv2.setUseOptimized(True)
		cv2.setNumThreads(4)
		im = cv2.imread('./dev/'+str(k+1)+'_.jpg')
		newHeight = 1000
		newWidth = int(im.shape[1]*1000/im.shape[0])
		im = cv2.resize(im, (newWidth, newHeight))
		box=segment(im)
		print (box)
		for i in range(0,len(box)):
			tmp = box[i]
			region=im[tmp[1]:tmp[3],tmp[0]:tmp[2]]
			#cv2.line(im,(tmp[0],tmp[1]),(tmp[2],tmp[3]),(255,255,0),2)
			cv2.imwrite(str(k+1)+'_'+str(i)+'.jpg',region)
			#cv2.imwrite(str(k+1)+'______ .jpg', im)
			
