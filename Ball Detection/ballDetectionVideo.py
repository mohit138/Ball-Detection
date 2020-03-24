import cv2
import imutils
import argparse
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

cap=cv2.VideoCapture(args["video"])
#cap = cv2.VideoCapture("E:\cv_stuff\Ball Detection and  Tracing\ball_tracking_example.mp4")
#cap=cv2.VideoCapture(0)
w=cap.get(3)
h=cap.get(4)
print("{}    {}".format(w,h))
w=int(.2*w)
h=int(.2*h)
print("{}    {}".format(w,h))

if (cap.isOpened()==False):
    print("Error opening video file")

while(cap.isOpened()):
    
    
    ret, frame= cap.read()
    frame = cv2.resize(frame,(w,h))    
    
    if ret == True:
        frame = cv2.resize(frame,(w,h))    
        #cv2.imshow("frame",frame)
        blur = cv2.GaussianBlur(frame,(9,9),0)
        #cv2.imshow("blur",blur)
        gray=cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#        cv2.imshow("gray",gray)
#        thresh=cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV)[1]
#        cv2.imshow("thresh",thresh)
#        edged=cv2.Canny(blur,20,100)
#        cv2.imshow("edged",edged)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#        cv2.imshow("hsv",hsv)
        
        lower_red = np.array([ 134, 135, 100])
        upper_red = np.array([215, 215, 256])
        
        mask = cv2.inRange(hsv, lower_red, upper_red)
        #cv2.imshow("mask",mask)
        mask = cv2.dilate(mask, None, iterations=10)
        #cv2.imshow("mask",mask)
        mask = cv2.erode(mask, None, iterations=10)
        #cv2.imshow("mask",mask)


        # using making contours, not so accurate, but can work


        # used moment to obtain the center of ball
        """
        cnt,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       # cnts = imutils.grab_contours(cnts)
        output=frame.copy()
        c = max(cnt, key= cv2.contourArea)
        hull=cv2.convexHull(c)
        cv2.drawContours(output, hull, -1, (0,255,0),3)
        cv2.imshow("output",output)

        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        print("{}    {}".format(cx,cy))
        cv2.circle(frame, (cx,cy),5,(0,255,0),-1)
        cv2.circle(frame, (cx,cy),44,(0,255,0),2) # ball
        cv2.imshow("ball",frame)
        """

        # used direct function to obtain the centre
        cnt,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnt, key= cv2.contourArea)
        hull=cv2.convexHull(c)
        (cx,cy),radius = cv2.minEnclosingCircle(hull)
        center=(int(cx),int(cy))
        radius=int(radius)
        cv2.circle(frame, center,5,(0,255,0),-1) #center of ball
        cv2.circle(frame, center,radius,(0,255,0),2) # ball
        cv2.imshow("ball",frame)
#        cv2.drawContours(output, cnts, -1, (0,255,0),3)
#        cv2.imshow("output",output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
