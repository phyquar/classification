import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2.cv as cv
from time import sleep
from sklearn.svm import LinearSVC

def capture_camera(size=None):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # frame = np.copy(tframe)
        # frame = frame.tolist()
        #print np.shape(frame)

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow('camera capture', frame)
        cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cimg = cv2.medianBlur(cimg, 3)
        cimg = cv2.GaussianBlur(cimg, (5, 5), 0)
        # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
        # _, coins_binary = cv2.threshold(cimg, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # coins_binary = cv2.bitwise_not(coins_binary)
        circles = cv2.HoughCircles(cimg, cv.CV_HOUGH_GRADIENT, 2, 10, param1=40, param2=60, minRadius=10, maxRadius=50)
        # print(circles)
        if circles is None:
            continue
        circles = np.uint16(np.around(circles))
        radius = np.average(circles[0, :, 2])  # mean radius of circles
        coordinates = circles[0, :][:, 0:2]

        delete = np.ones(np.shape(coordinates)[0], dtype=bool)
        for i in range(np.shape(coordinates)[0]):
            for j in range(i):
                if np.sqrt((float(coordinates[i, :][0]) - float(coordinates[j, :][0]))**2 + (float(coordinates[i, :][1]) - float(coordinates[j, :][1]))**2) < radius*1.5:
                    delete[i] = 0
        coordinates = coordinates[delete, :]
        circles = circles[:, delete, :]
        red_circles=np.array([],dtype=np.int)
        blue_circles=np.array([],dtype=np.int)
        #print circles
        for i in circles[0,:]:
            #print i
            pixel_value=frame[i[1],i[0]]
            #print pixel_value
            if pixel_value[0]>100. and pixel_value[1]<50. and pixel_value[2]<50.:
                blue_circles=np.append(np.array(blue_circles),np.array(i),axis=0)
            elif pixel_value[0]<50. and pixel_value[1]<50. and pixel_value[2]>80:
                red_circles=np.append(np.array(red_circles),np.array(i),axis=0)

        red_circles=red_circles.reshape((red_circles.size/3,3))
        blue_circles=blue_circles.reshape((blue_circles.size/3,3))
        
        if red_circles.size==0 or blue_circles.size==0:
            continue
        #print red_circles
        #print blue_circles

        training_data_location=np.append(red_circles[:,:],blue_circles[:,:],axis=0)
        #print training_data_location

        training_data_label=np.array([-1]*red_circles.shape[0]+[1]*blue_circles.shape[0])
        #print training_data_label



        svm_model = LinearSVC(C=1.0,intercept_scaling=np.shape(frame)[0],random_state=0)
        svm_model.fit(training_data_location[:,0:2],training_data_label)
        #print circles[0, :]
        number = np.shape(circles[0, :])[0]

        j = 0
        for i in training_data_location[:]:
            #print 'i', i
            if j < red_circles.shape[0]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 2)
            else:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
            j = j + 1
        sleep(0.1)
        #y=a*x+b
        a=-1*svm_model.coef_[0][0]/svm_model.coef_[0][1]
        b=-1*svm_model.intercept_/svm_model.coef_[0][1]
        if b>0:
            if a*np.shape(frame)[1]+b<np.shape(frame)[0]:
                cv2.line(frame,(0,b),(np.shape(frame)[1],a*np.shape(frame)[1]+b),(0,255,0),3)
            else:
                cv2.line(frame,(0,b),((np.shape(frame)[0]-b)/a,np.shape(frame)[0]),(0,255,0),3)
        else:
            if a*np.shape(frame)[1]+b<np.shape(frame)[0]:
                cv2.line(frame,(-1*b/a,0),(np.shape(frame)[1],a*np.shape(frame)[1]+b),(0,255,0),3)
            else:
                cv2.line(frame,(-1*b/a,0),((np.shape(frame)[0]-b)/a,np.shape(frame)[0]),(0,255,0),3)

        cv2.imshow('detected circles', frame)


        k = cv2.waitKey(1)
        if k == 27:
            break

        

    cap.release()
    cv2.destroyAllWindows()

capture_camera()