#!/usr/bin/python
# -*- coding: utf-8 -*-
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough. 
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import datetime

import os
import dlib
import glob
from skimage import io
import cv2
import numpy as np
import pickle
from time import sleep

print dlib.__version__
print dlib.__file__
#print dlib.__path__
from distutils.sysconfig import get_python_lib
print(get_python_lib())

win = dlib.image_window()
class FaceRecognizer:
    def __init__(self,
                 predictor_path="shape_predictor_68_face_landmarks.dat",
                 face_rec_model_path="dlib_face_recognition_resnet_model_v1.dat",
                 people_descriptor_file_name="people_descriptor.pk"):
	print "begin construtor"
        self.detector = dlib.get_frontal_face_detector()
	print "begin get_frontal_face_detector"
        self.sp = dlib.shape_predictor(predictor_path)
	print "begin face_recognition_model_v1"
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
	print "begin people_descriptor_file_name"
        self.people_descriptor=pickle.load(open(people_descriptor_file_name,"rb"))
	print "endconstrutor"
        self.verbose=False
    def recognizer(self,img):
        win.clear_overlay()
        win.set_image(img)
        
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self.detector(img, 1)
        nFaces=len(dets)
        if self.verbose:
            print("Number of faces detected: {}".format(nFaces))
        
        faces={}
        # Now process each face we found.
        for k, d in enumerate(dets):
            if self.verbose:
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = self.sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            #win.clear_overlay()
            win.add_overlay(d);
            win.add_overlay(shape)
        
            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people.  He we just print
            # the vector to the screen.
            face_descriptor = self.facerec.compute_face_descriptor(img, shape);
            minP=1.0
            minK=""
            for k in self.people_descriptor:
                t=0
                for d in self.people_descriptor[k]:
                    dif=np.array(d)-np.array(face_descriptor)
                    t+=np.linalg.norm(dif)
                p=t/len(self.people_descriptor[k])
                if p<minP:
                    minP=p
                    minK=k
            faces[minK]=(minP,d,shape)
        return faces

import thread
captureProcessed=False

def capture():
    global captureProcessed,img,img0
    while True:
        #print captureProcessed
        ret,img0=cap0.read()
#         if not ret:
#             print "Error al capturar imagen"
#         else:
#             if captureProcessed==True:
#                 #print "Capturando"
#                 img=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
#                 capturedImage=True
#                 captureProcessed=False

def nombreDia():
    h=datetime.datetime.now().hour
    if h>14 and h<21:
        return "Buenas tardes"
    if h>=21 or h<6:
        return "Buenas noches"
    if h>=6 and h<=14:
        return "Buenos días"
    
fr=FaceRecognizer()  
f="~/face_recognition/marisa_cea/face2.jpg"
print("Processing file: {}".format(f))
#img = io.imread(f)
#cap0 = cv2.VideoCapture("http://192.168.43.240:8080/video")
cap0 = cv2.VideoCapture(0)
ret,img0=cap0.read()
if not ret:
    print "Error al capturar imagen"
else:
        img=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
thread.start_new_thread(capture,())
while True:
    if not captureProcessed:
        #now img is global var
        img=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
        faces=fr.recognizer(img)
        for name in faces:
            print name,faces[name]
            name=name.replace("_"," ")
            command='~/di.sh "Hola. %s."'%nombreDia()
            os.system(command)
        #captureProcessed=True

    




