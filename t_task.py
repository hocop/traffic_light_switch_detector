import numpy as np
import cv2
from math import *
import requests
import time
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

view = False
erosion = 6
num_prev = 10
path = 'testset/'
foutName = 'res8'
minArea = 16

# boundaries
lower_green1 = np.array([80,100,100])#80,100,100
upper_green1 = np.array([95,255,255])#95,,
lower_green2 = np.array([55,100,100])#55,100,100
upper_green2 = np.array([105,255,255])#105,,
lower_red1 = np.array([0,80,100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([145,80,100])
upper_red2 = np.array([180,255,255])
lower_red3 = np.array([10,80,100])
upper_red3 = np.array([20,255,255])


def toBin(mask):
	mask = cv2.dilate(mask, None, iterations=erosion)
	mask = cv2.erode(mask, None, iterations=erosion)
	return mask

def xyr(c):
	M = cv2.moments(c)
	x = int(M['m10'] / (M['m00']+1))
	y = int(M['m01'] / (M['m00']+1))
	r = int(sqrt(M['m00']/pi))
	return x, y, r

def shape(c, r, l): # check circle
	area = pi*r**2
	if area < minArea:
		return 'small'
	if abs(l - 2*pi*r) > l/6:
		return 'blet, ovalny'
	return 'norm'

def catchSwitch(c1, c2, allPrev):
	crs1, cgs1 = c1
	crs2, cgs2 = c2
	cgs1 = [c for c in cgs1]
	for ap in allPrev:
		cgs1 += ap[1]
	
	# find disappeared red contours
	oldR = []
	for cr1 in crs1:
		x1, y1, r1 = xyr(cr1)
		# check shape
		l = cv2.arcLength(cr1,True)
		if shape(cr1, r1, l) != 'norm': # not close to circle
			continue
		# check disappear
		intersection = False
		for cr2 in crs2:
			x2, y2, r2 = xyr(cr2)
			if (x1-x2)**2 + (y1-y2)**2 < (r1+r2)**2:
				intersection = True
				break
		if not intersection:
			oldR += [cr1]
	
	# find new green contours
	newG = []
	for cg2 in cgs2:
		x2, y2, r2 = xyr(cg2)
		# check shape
		l = cv2.arcLength(cg2,True)
		if shape(cg2, r2, l) != 'norm': # not close to circle
			continue
		# check if reds are near
		closeRed = False
		for oR in oldR:
			xr, yr, rr = xyr(oR)
			if (x2-xr)**2 + (y2-yr)**2 < 36 * (r2+rr)**2:
				closeRed = True
				break
		if not closeRed:
			continue
		# check appear
		intersection = False
		for cg1 in cgs1:
			x1, y1, r1 = xyr(cg1)
			if (x1-x2)**2 + (y1-y2)**2 < 2*(r1+r2)**2:
				intersection = True
				break
		if not intersection:
			newG += [cg2]
	
	# find switch
	for r in oldR:
		xr, yr, rr = xyr(r)
		for g in newG:
			xg, yg, rg = xyr(g)
			meanr = (rr+rg)/2
			if abs(xr-xg) < meanr * 0.66: # x-column
				if float(max(rr,rg))/min(rr,rg) < 2: # sizes match
					if yg-yr > meanr / 2: # y upper
						if yg-yr < meanr * 6: # y lower
							return True, r, g
	return (False,)

def solve(fname):
	f = path + fname
	cap = cv2.VideoCapture(f)
	prev = range(num_prev) # previous frames info
	num = 0 # current frame number
	foundSwitches = []
	foundFrame = -1
	
	# start
	while(cap.isOpened()):
		# Time measure
		start = time.time()
	
		# Capture frame-by-frame
		ret, frame0 = cap.read()
		if not ret:
			break
		num += 1
		frame0 = cv2.medianBlur(frame0, 5)
#		frame0 = cv2.medianBlur(frame0, 5)
		frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
	
		# red mask
		red1 = cv2.inRange(frame, lower_red1, upper_red1)
		red2 = cv2.inRange(frame, lower_red2, upper_red2)
		red3 = cv2.inRange(frame, lower_red3, upper_red3)
		# green mask
		green1 = cv2.inRange(frame, lower_green1, upper_green1)
		green2 = cv2.inRange(frame, lower_green2, upper_green2)
		
		# detect contours
		r2,contRed1,h=cv2.findContours(red1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		r2,contRed2,h=cv2.findContours(red2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		r2,contRed3,h=cv2.findContours(red3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		g2,contGreen1,h=cv2.findContours(green1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		g2,contGreen2,h=cv2.findContours(green2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		contRed = contRed1 + contRed2 + contRed3
		contGreen = contGreen1 + contGreen2
		
		# pre-process contours
		for i in range(len(contRed)):
			contRed[i] = cv2.convexHull(contRed[i])
		for i in range(len(contGreen)):
			contGreen[i] = cv2.convexHull(contGreen[i])
		prev = [(contRed, contGreen)] + prev[0:]
	
		if num < 6:
			continue
	
		# search for switches
		found = False
		maxI = min(5,num_prev/2,num/2)
		for i in range(num_prev)[1: maxI]:
			found = catchSwitch(prev[i], prev[0], prev[i+1:maxI])
			if found[0]:
				foundFrame = num
				foundSwitches += [found[1]]
				break
		if foundFrame > 0 and not view:
			break

		# Display the resulting frame
		if view:
			gray = frame0
			cv2.drawContours(gray, contRed, -1, (0,0,255), 3)
			cv2.drawContours(gray, contGreen, -1, (0,255,0), 3)
			cv2.drawContours(gray, foundSwitches, -1, (255,0,0), 16)
			cv2.imshow('frame', gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	
		# Time
		end = time.time()
		#print 'frame took', end - start
	cap.release()
#	print fname, foundFrame
	fout = open(foutName, 'a')
	fout.write(fname + ' ' + str(foundFrame) + '\n')
	fout.close()

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles.sort()
#onlyfiles = ['akn.060.018.left.avi']

# begin program
fout = open(foutName, 'w')
fout.close()
if view:
	for fname in onlyfiles:
		solve(fname)
else:
	pool = Pool()
	results = [pool.apply_async(solve, [fname]) for fname in onlyfiles]
	for r in results:
		r.get()

cv2.destroyAllWindows()



















