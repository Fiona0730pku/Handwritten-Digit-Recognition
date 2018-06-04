import sys
import cv2
import numpy as np
 
for k in range(10):
	cv2.setUseOptimized(True)
	cv2.setNumThreads(4)
	im = cv2.imread('./dev/'+str(k+1)+'_.jpg')
	# resize image
	newHeight = 1000
	newWidth = int(im.shape[1]*1000/im.shape[0])
	im = cv2.resize(im, (newWidth, newHeight))

	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(im)
	ss.switchToSelectiveSearchQuality()
 
	# run selective search segmentation on input image
	rects = ss.process()
	rec = []
	imOut = im.copy()
	for i, rect in enumerate(rects):
		x, y, w, h = rect
		if w >= h * 2:
			continue
		if w <= 5:
			continue
		if w >= newWidth*0.9 or h>=newHeight*0.9:
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
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	for i in range(len(rec)):
		if ava[i]:
			x, y, w, h = rec[i]
			p = [0 for i in range(w-x)]
			rmx = [0 for i in range(w-x)]
			for j in range(w-x):
				lo = 0
				hi = 0
				for l in range(h-y):
					if im[j+x, l+y]<= 127:
						lo = l
						break
				for l in range(h-y):
					if im[j+x, l+y]<= 127:
						hi = l+1

				p[j] = hi-lo
			'''rmx[w-x-1] = p[w-x-1]
			for j in range(1, w-x):
				rmx[w-x-1-j] = np.max(p[w-x-1-j], rmx[w-x-j])

			segpoint = []
			segpoint.append(x)
			lmx = 0
			for j in range(w-x):
				lmx = np.max(lmx, p[j])
				if lmx - p[j] >= 20 '''
			print(p)

			cv2.rectangle(imOut, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)
	#while True:
		# show output
	cv2.imwrite('./dev/'+str(k+1)+'__.jpg', imOut)
	# close image show window
	#cv2.destroyAllWindows()