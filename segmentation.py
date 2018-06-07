import sys
import cv2
import numpy as np
 
def segment(im):
	kernel = np.ones((5,5),np.uint8)
	im = cv2.erode(im,kernel,iterations = 1)
		
		# resize image

	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(im)
	ss.switchToSelectiveSearchQuality()
	height, width, d = im.shape
	 
		# run selective search segmentation on input image
	rects = ss.process()
	rec = []
	box = []
	########################################################################
	GrayImage=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)   #将BGR图转为灰度图
	ret,thresh1=cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY) 
	#########################################################################
		#imOut = im.copy()
	for i, rect in enumerate(rects):
		x, y, w, h = rect
		if w >= h * 2:
			continue
		if w <= 5:
			continue
		if w >= width*0.9 or h>=height*0.9:
			continue
		rec.append((x,y,x+w,y+h))
			#cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
	ava = [True for i in range(len(rec))]
	for i in range(len(rec)):
		xi, yi, wi, hi = rec[i]
		for j in range(len(rec)):
			if i==j:
				continue
			xj, yj, wj, hj = rec[j]
			if xi >= xj and yi>=yj and wi<=wj and hi<=hj:
				ava[i]=False
				break
	#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		#box=[]
	for i in range(len(rec)):
		if ava[i]:
			x, y, w, h = rec[i]
			box.append([x,y,w,h])
			#cv2.line(im,(x,y),(w,h),(255,255,0),2)
	#################################################################################
			'''m = 0
			a = [0 for z in range(0, w)] 
				#记录每一列的波峰
			for j in range(x,w): #遍历一列 
				for i in range(y,h):  #遍历一行
					if  thresh1[i,j]==0:  #如果改点为黑点
						a[j]+=1  		#该列的计数器加一计数 
				if a[j] > m:
					m = a[j]        
				
			hight = m/3
			left = x
			right = w
				
			for j in range(x,w):
				if a[j]<=m/4:
					left+=1
				else:
					#box.extend([x,y,w,h])
					break
						
			for j in range(w,x):
				if a[j]<=m/4:
					right-=1
				else:
					#box.extend([x,y,w,h])
					break

			num = 0
			pre = y
			for j in range(left+1, right):
				if a[j] <= m/4:
					minm = m
					for i in range(j, right):
						if a[i] <= minm:
							minm = a[i]
							num = i
						if a[i] > m/2:
							#cv2.line(imOut,(num,y),(num,h),(0,255,0),2)
							#box.extend([pre,y,num,h])
							pre = num
							j = i
							break
			#box.extend([pre,y,w,h])
	#################################################################################
			#cv2.rectangle(imOut, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)
			#print (rec)'''
	return box
		#while True:
			# show output
		#cv2.imwrite('./dev/'+str(k+1)+'__ .jpg', imOut)
		# close image show window
		#cv2.destroyAllWindows()
