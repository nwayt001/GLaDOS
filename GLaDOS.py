# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:26:05 2017

Genetic Lifeform and Disk Operating System
(GLaDOS)

@author: Waytowich

"""

from pygame import mixer # Load the required library
import time
import httplib2
from BeautifulSoup import BeautifulSoup, SoupStrainer
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

class GLaDOS(object):
    def __init__(self):
        # initialize audio mixer
        mixer.init()
        
        # load audio library
        self.load_audio_library()
        
        # play welcome msg
        self.speak(self.wakeupMsg)
        
    def load_audio_library(self):
        self.wakeupMsg = 'D:\GLaDOS\\audio_library\chellgladoswakeup01.mp3'
        self.library_dir = 'audio_library\\'
        self.audiofiles  = [f for f in listdir(self.library_dir) if isfile(join(self.library_dir, f))]        
        
    def speak(self,msg):
        mixer.music.load(msg)
        mixer.music.play()
        
    def stop_speak(self):
        mixer.music.stop()
    
    def run(self):
        while True:
            time.sleep(60)
            self.speak_random_msg()

    def speak_random_msg(self):
        idx = np.random.randint(0,len(self.audiofiles)-1)
        msg_file = join(self.library_dir, self.audiofiles[idx])
        self.speak(msg_file)
       
class GLaDOS_Interactive_Eye(GLaDOS):
    def __init__(self, detect_faces = False):
        super(GLaDOS_Interactive_Eye, self).__init__()
        
        self.just_said_something = False
        self.time_since_speaking = 0
        self.detection_threshold = 0.4
        
        # initialzie GLaDOS's eye (rgb camera)
        self.eye = cv2.VideoCapture(0)
        
        # test that eye is functional 
        if self.eye.isOpened(): # try to get the first frame
            self.rval, frame = self.eye.read()
            print('GLaDOS eye initiated')
        else:
            self.rval = False
            print('GLaDOS eye failed to initiate')
          
        # initialize a face detector for GLaDOS
        self.faceDetector = FaceDetector()
        self.detect_faces = detect_faces
        self.face_detection_msgs = []
        self.face_detection_msgs.append('D:\GLaDOS\\audio_library\\taunt_big_wave05.mp3')
        self.face_detection_msgs.append('D:\GLaDOS\\audio_library\\taunt_small_wave01.mp3')
        self.just_detected_face = False
        self.time_since_face_detection = 0
        
    # main runnable
    def run(self):
        
        while self.rval:
            self.rval, frame1 = self.eye.read()
            time.sleep(0.02)
            self.rval, frame2 = self.eye.read()
            key = cv2.waitKey(50)            
            fdiff = frame1.astype('float32') - frame2.astype('float32')
            avg = fdiff.mean()
            
            # detect any faces
            if self.detect_faces:
                faces = self.faceDetector.detectFaces(frame1)
                if faces and not self.just_detected_face:
                    print('detected a face')
                    self.speak(self.face_detection_msgs[np.random.randint(0,2)])
                    self.just_detected_face = True
                    self.time_since_face_detection = time.time()
                    self.just_said_something = True
                    self.time_since_speaking = time.time()
                    
                if time.time() - self.time_since_face_detection > 10:
                    self.just_detected_face = False
                    
            # detect any motion of any kind
            if avg >= self.detection_threshold:
                print('motion detected')
                
                # check when the last time GLaDOS spoke something
                if time.time() - self.time_since_speaking > 50:
                    self.just_said_something  = False
                
                # speak
                if not self.just_said_something:
                    self.speak_random_msg()
                    self.just_said_something = True
                    self.time_since_speaking = time.time()
            
            if key ==27:
                break
        
        # terminate on exit
        self.terminate()
        
    
    def terminate(self):
        print('terminating...good bye.')
        self.eye.release()
        exit()
        
# Peripheral face detection class    
class FaceDetector(object):
    def __init__(self, run_vis = False):
        
        self.cascPath = "D:\opencv\\opencv\\build\\etc\\haarcascades\\"
        
        self.cascades =[]
        self.cascades.append("haarcascade_frontalface_default.xml")
        self.cascades.append("haarcascade_eye_tree_eyeglasses.xml")
        self.cascades.append("haarcascade_profileface.xml")
        
        # Create the haar cascade
        self.faceClassifier = []
        for cascade in self.cascades:
            self.faceClassifier.append(cv2.CascadeClassifier(os.path.join(self.cascPath, cascade)))
        
        if run_vis:
            self.realTimeVis()
        
    def realTimeVis(self):
        # get image of face
        cv2.namedWindow("FaceDetector")
        self.vc = cv2.VideoCapture(0)
        rval, image = self.vc.read()

        while rval:        
            rval, image = self.vc.read()
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
            # detect faces and other features in the image
            for classifier in self.faceClassifier:
                objects = classifier.detectMultiScale(
                    img,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
                if len(objects) > 0:
                    #print "Found {0} objects!".format(len(faces))
                    # Draw a rectangle around the detected objects
                    for (x, y, w, h) in objects:
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
           
            cv2.imshow("FaceDetector", image)
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        self.vc.release()
        cv2.destroyWindow("FaceDetector")        
        
    def detectFaces(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detected_face = False
        
        # detect faces in the image
        for classifier in self.faceClassifier:
            faces = classifier.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                detected_face = True
                
        return detected_face
        
        
# Peripheral video capture Class
class VideoCapture(object):
    def __init__(self):
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        vc.release()
        cv2.destroyWindow("preview")
        

# Peripheral library downloader class
class LibraryDownloader(object):
    def __init__(self):
        http = httplib2.Http()
        status, response = http.request('http://www.portal2sounds.com/#w=GLaDOS')
        
        links = BeautifulSoup(response, parseOnlyThese=SoupStrainer('a'))
        numlinks = len(links)
        for link in links:
            if link.has_key('href'):
                l = link['href']
                print link['href']
            

# Main entry point
if __name__ == '__main__':
    # main GLaDOS
    #self = GLaDOS()
    #self.run()
    
    # interactive GLaDOS with eye
    self = GLaDOS_Interactive_Eye()
    self.run()

    self = FaceDetector(run_vis = True)
