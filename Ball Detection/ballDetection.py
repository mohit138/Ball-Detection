# ball detect

import cv2
import imutils
import argparse
import numpy as np
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
(h,w,d)=image.shape

w=int(.15*w)
h=int(.15*h)
image = cv2.resize(image,(w,h))
cv2.imshow("Image",image)



blur = cv2.GaussianBlur(image,(9,9),0)
#cv2.imshow("blur",blur)


hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#cv2.imshow("hsv",hsv)


lower_red = np.array([ 134, 135, 100])
upper_red = np.array([ 215, 215, 256])

mask = cv2.inRange(hsv, lower_red, upper_red)
#cv2.imshow("mask",mask)


mask = cv2.dilate(mask, None, iterations=12)
cv2.imshow("mask_dilated",mask)
mask = cv2.erode(mask, None, iterations=12)
cv2.imshow("mask_eroded",mask)



"""
cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
output=image.copy()
cv2.drawContours(output, cnts, -1, (0,255,0),3)
cv2.imshow("output",output)
"""

# below are various ways to use

# 1. find approx center of contour,( if only a single contour is detected.) 
"""
cnt,hierarchy = cv2.findContours(mask.copy(), 1,2)
print(cnt)
cntr=cnt[0]
M = cv2.moments(cntr)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print("{}    {}".format(cx,cy))
cv2.circle(image, (cx,cy),5,(0,255,0),-1) #center of ball
cv2.circle(image, (cx,cy),37,(0,255,0),2) # ball
cv2.imshow("ball",image)
"""

"""
# 2. finding center of largest contour

output = image.copy()
cnt,hierarchy = cv2.findContours(mask.copy(), 1,2)

c = max(cnt, key= cv2.contourArea)

M = cv2.moments(c)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv2.contourArea(c)
radius=math.sqrt(area/3.1415)
print("{}    {}".format(cx,cy))
cv2.circle(image, (cx,cy),5,(0,255,0),-1) #center of ball
cv2.circle(image, (cx,cy),int(radius),(0,255,0),2) # ball
cv2.imshow("ball",image)
"""

#3. maximum area is made convex from all sides. using convex hull
"""
output = image.copy()
cnt,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = max(cnt, key= cv2.contourArea)

hull=cv2.convexHull(c)
cv2.drawContours(output, hull, -1, (0,255,0),3)
cv2.imshow("contour",output)
M = cv2.moments(hull)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
area = cv2.contourArea(c)
radius=math.sqrt(area/3.1415)
print("{}    {}".format(cx,cy))
cv2.circle(image, (cx,cy),5,(0,255,0),-1) #center of ball
cv2.circle(image, (cx,cy),int(radius),(0,255,0),2) # ball
cv2.imshow("ball",image)
"""

#4. maximum contour is used, to obtain approx radius and center
output = image.copy()
cnt,hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = max(cnt, key= cv2.contourArea)
hull=cv2.convexHull(c)
cv2.drawContours(output, hull, -1, (0,255,0),3)
cv2.imshow("output",output)

(cx,cy),radius = cv2.minEnclosingCircle(hull)
center=(int(cx),int(cy))
radius=int(radius)
cv2.circle(image, center,5,(0,255,0),-1) #center of ball
cv2.circle(image, center,radius,(0,255,0),2) # ball
cv2.imshow("ball",image)


cv2.waitKey(0)

