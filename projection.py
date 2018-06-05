import cv2  
import numpy  
img = cv2.imread('s1.png',cv2.COLOR_BGR2GRAY)  
height, width = img.shape[:2]  
#print (height, width)  
#resized = cv2.resize(img, (2*width,2*height), interpolation=cv2.INTER_CUBIC)  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
(_, thresh) = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY) 
#cv2.imshow('t',thresh)
#使文字增长成块  
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))#形态学处理，定义矩形结构  
#closed = cv2.erode(thresh, None, iterations = 7)  
#cv2.imshow('erode',closed)  
#height, width = closed.shape[:2]  
#print height, width  
z = [0]*width  
v = [0]*height 
hfg = [[0 for col in range(2)] for row in range(height)]  
lfg = [[0 for col in range(2)] for row in range(width)]  
box = [0,0,0,0]  
#水平投影  
a = 0  
emptyImage1 = numpy.zeros((height, width, 3), numpy.uint8)   
for y in range(0, width):  
	for x in range(0, height):  
		cp = thresh[x,y]   
		#if np.any(closed[y,x]):  
		if cp == 0:  
			a = a + 1  
		else :  
			continue  
	z[y] = a  
	a = 0  
#根据水平投影值选定行分割点  
inline = 1  
start = 0  
j = 0  
for i in range(0,width):  
	if inline == 1 and z[i] >= 5 :  #从空白区进入文字区  
		start = i  #记录起始行分割点  
		#print (i)  
		inline = 0  
	elif (i - start > 3) and z[i] < 5 and inline == 0 :  #从文字区进入空白区  
		inline = 1  
		hfg[j][0] = start - 2  #保存行分割位置  
		hfg[j][1] = i + 2  
		j = j + 1  

	#垂直分割点  
	incol = 1  
	start1 = 0  
	j1 = 0 
	for p in range(0,j): 
		#print (hfg[p][0],hfg[p][1]) 		
		cv2.rectangle(img, (hfg[p][0],0), (hfg[p][1],height), (255,0,0), 2)              
cv2.imshow('result', img)  
x=cv2.waitKey(0) 
while x!=113:	#键盘输入q
	x=cv2.waitKey(0) 