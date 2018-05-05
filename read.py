import cv2

def read(filename):
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	return img

def write(filename, img):
	cv2.imwrite(img, filename)