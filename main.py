import numpy as np
import cv2 as cv
import os
from keras.models import load_model
import time
import pyfirmata
from time import sleep


PORT = "/dev/tty.usbserial-1420"
board = pyfirmata.Arduino(PORT)
button1 = board.get_pin('d:2:i')
button2 = board.get_pin('d:3:i')
LED_0 = 11
LED_1 = 12

it = pyfirmata.util.Iterator(board)
it.start()
servo1 = board.get_pin('d:9:s')
servo2 = board.get_pin('d:10:s')

#board.pass_time(.2)
servo1.write(100)

board.pass_time(.5)
servo2.write(60)
print('servo ok!')

button1.enable_reporting()

cap = cv.VideoCapture(1)
videoWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
videoHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
model = load_model('human_detection_model.h5')
choicesModel = load_model('rs_model.h5')

def paddingWithBlackColor(image, minx, miny, maxx, maxy):
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            if ((width < minx or width > maxx) or (height < miny or height > maxy)):
                image[height, width, 0] = 0
                image[height, width, 1] = 0
                image[height, width, 2] = 0

    return image
face_cascade = cv.CascadeClassifier('datasets/haarcascade_frontalface_default.xml')
def DrawFace(img):
    global frameCount, label, stopCount
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        paddingImage = paddingWithBlackColor(img, x, y, x+w, y+h)
        #cropImage = img[y: y+h, x: x+w]
        #print("hoffset = " + str(h) + ", woffset = " + str(w))
        
        cv.imshow('video', paddingImage)

def CalculateFaceArea(img):
    global frameCount, label, stopCount
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        if w > 70 and h > 70:
            return True

totalFrameArray = []

ct = 0
state = 0
def Blink(state, ct):
    board.digital[LED_0].write(state)
    board.digital[LED_1 ].write(not state)
    if ct%10==0: 
        state= not state
    ct+=1

def dropItem0():
    for i in range(60,0,-1):
        servo2.write(i)
        print(i)
        board.pass_time(.1)
    for i in range(0,60,1):
        servo2.write(i)
        print(i)
        board.pass_time(.1)
    
    return 0

def dropItem1():
    for i in range(100,200,1):
        servo1.write(i)
        print(i)
        board.pass_time(.1)
    for i in range(200,100,-1):
        servo1.write(i)
        print(i)
        board.pass_time(.1)
    
    return 0

def activateSystem():
    resultArray = np.zeros((1,4), dtype = np.int32)
    global state, ct
    #waiting for person coming in...
    while(True):

        Blink(state, ct) 
        ret, frame = cap.read()
        resizeFrame = cv.resize(frame, (int(videoWidth/4), int(videoHeight/4)))

        hasPerson = CalculateFaceArea(resizeFrame)
        if hasPerson:
            break
    #person has arrived, start detecting
    timeStart = time.time()
    ctWait = 0
    while(True):
        ctWait = ctWait + 1

        ret, frame = cap.read()
        resizeFrame = cv.resize(frame, (int(videoWidth/4), int(videoHeight/4)))

        DrawFace(resizeFrame)
        resizeFrame = resizeFrame / 255

        imagenpy = resizeFrame.reshape((1, resizeFrame.shape[0], resizeFrame.shape[1], resizeFrame.shape[2]))
        result = model.predict_classes(imagenpy)
        print(result)

        board.digital[LED_0].write(1)
        board.digital[LED_1].write(1)
        

        cv.imshow('video', resizeFrame)
        if cv.waitKey(1) == ord('q'):
            break

        totalFrameArray.append(result)
        if time.time() - timeStart > 10:
            #統計結果
            for i in range(len(totalFrameArray)):
                if totalFrameArray[i] == 0:
                    resultArray[0,0] = resultArray[0,0] + 1
                if totalFrameArray[i] == 1:
                    resultArray[0,1] = resultArray[0,1] + 1
                if totalFrameArray[i] == 2:
                    resultArray[0,2] = resultArray[0,2] + 1
                if totalFrameArray[i] == 3:
                    resultArray[0,3] = resultArray[0,3] + 1
            break
    ImportModel(resultArray)
    while(True):
        board.pass_time(.1)
        if (str(button1.read())=='False'):
            print(dropItem1())
            break
        if (str(button2.read())=='False'):
            print('2')
            print(dropItem0())
            break


def ImportModel(resultArray):
    rs_result = choicesModel.predict([resultArray])
    print(rs_result)

    if rs_result[0][0] == 1:
        print("recommandate 0")
        board.digital[LED_0].write(1)
        board.digital[LED_1].write(0)
         
    if rs_result[0][1] == 1:
        print("recommandate 1")
        board.digital[LED_0].write(0)
        board.digital[LED_1].write(1)

activateSystem()
