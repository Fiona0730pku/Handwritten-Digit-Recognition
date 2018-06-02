import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('10.jpg') 
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
edges = cv2.Canny(img,20,20)
im = img.copy()
plt.subplot(121),plt.imshow(im)
plt.xticks([]),plt.yticks([])
#hough transform
rows,cols = img.shape[:2]
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


ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=30, min_size=100)

segment = segmentator.processImage(img)
seg_image = np.zeros(img.shape, np.uint8)

for i in range(np.max(segment)):
  y, x = np.where(segment == i)

  top, bottom, left, right = min(y), max(y), min(x), max(x)
  cv2.rectangle(img, (left, bottom), (right, top), (0, 255, 0), 1)


plt.subplot(122),plt.imshow(img)
plt.xticks([]),plt.yticks([])
plt.show()