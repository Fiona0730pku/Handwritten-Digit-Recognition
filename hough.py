import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(10):
	img = cv2.imread('./dev/'+str(i+1)+'.jpg', 0)
	rows,cols = img.shape[:2]
	for i in range(rows):
		for j in range(cols):
			if img[i][j] > 250:
				img[i][j] = 255
			else:
				img[i][j] = 0

	#ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	edges = cv2.Canny(img,20,20)
	im = img.copy()
	#hough transform
	lines = cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength=10,maxLineGap=50)
	lines1 = lines[:,0,:]#提取为二维
	sp = img.shape
	x = sp[1] + 1000 #width(colums) of image
	for x1,y1,x2,y2 in lines1[:]: 
		x1 = float(x1)
		x2 = float(x2)
		y1 = float(y1)
		y2 = float(y2)
		if x2-x1 != 0:
			k = (y2-y1)/(x2-x1)
			y1 = y1 - k*x1
			y2 = y1+k*cols
		x1 = int(x1)
		x2 = int(x2)
		y1 = int(y1)
		y2 = int(y2)
		cv2.line(img,(0,y1),(cols,y2),(255,255,255),2)

	img = img[550: rows, 0: cols]
	cv2.imwrite('./dev/'+str(i+1)+'_.jpg', img)