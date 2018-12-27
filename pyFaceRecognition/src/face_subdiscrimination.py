#!/usr/bin/python
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
import os
import numpy as np
import dlib
import glob
from skimage import io
from sklearn.manifold import TSNE
import cv2
import matplotlib.pyplot as plt

print dlib.__version__
print dlib.__file__
print dlib.__path__
from distutils.sysconfig import get_python_lib
print(get_python_lib())

#Mood expressions
normal=0
happy=1
shad=2
surprise=3

def distance(d,face_descriptor):
    #d=np.array(d)
    #face_descriptor=np.array(face_descriptor)
    #return np.sum(np.square(d - face_descriptor))
    #return d.dot(face_descriptor)
    dif=d-face_descriptor
    t=np.linalg.norm(dif)
    #t=np.sqrt(dif.dot(dif))
    return t

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_68_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path);

win = dlib.image_window()

#cap0 = cv2.VideoCapture(0)
# Now process all the images
#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#faces=[[None]*10]*7   #Bad way of init, it is the same raw 
faces=[ [ None for i in range(10)] for j in range(7)]

#ret,img0=cap0.read()
img0=cv2.imread('/home/francisco/Downloads/Facial-expressions-of-the-dataset-The-complete-dataset-is-composed-of-70-2D-photographs.jpg.png')
#img0=cv2.resize(img0, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
#ret,img1=cap1.read()
#if not ret:
#    print "Algo valla"
rows,cols,channels =img0.shape
rows_g=rows/7
cols_g=cols/10
print rows_g,cols_g,channels

#print("Processing file: {}".format(f))
#img = io.imread(f)
#cv2.imshow("0",img0)
#cv2.imshow("1",img1)
#k=cv2.waitKey(10)

img=cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
print img.shape

win.clear_overlay()
win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

# Now process each face we found.
for k, d in enumerate(dets):
    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #    k, d.left(), d.top(), d.right(), d.bottom()))
    center_row=(d.bottom()+d.top() )/2
    center_col=(d.right() +d.left())/2
    center_row_g=center_row / rows_g
    center_col_g=center_col / cols_g
    #print center_row,center_col,center_row_g,center_col_g
    #print center_row_g,center_col_g
    # Get the landmarks/parts for the face in box d.
    shape = sp(img, d)
    # Draw the face landmarks on the screen so we can see what face is currently being processed.
    #win.clear_overlay()
    win.add_overlay(d);
    win.add_overlay(shape)

    # Compute the 128D vector that describes the face in img identified by
    # shape.  In general, if two face descriptor vectors have a Euclidean
    # distance between them less than 0.6 then they are from the same
    # person, otherwise they are from different people.  He we just print
    # the vector to the screen.
    face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))
    #print "descriptor length=",np.linalg.norm(face_descriptor)
    faces[center_row_g][center_col_g]=np.array(facerec.compute_face_descriptor(img, shape))
    # It should also be noted that you can also call this function like this:
    #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100);
    # The version of the call without the 100 gets 99.13% accuracy on LFW
    # while the version with 100 gets 99.38%.  However, the 100 makes the
    # call 100x slower to execute, so choose whatever version you like.  To
    # explain a little, the 3rd argument tells the code how many times to
    # jitter/resample the image.  When you set it to 100 it executes the
    # face descriptor extraction 100 times on slightly modified versions of
    # the face and returns the average result.  You could also pick a more
    # middle value, such as 10, which is only 10x slower but still gets an
    # LFW accuracy of 99.3%.

print "Mean faces descriptor for each person"
meanPerson=[]
for person in range(10):
    mean=faces[0][person].copy()
    for mood in range(1,7):
        mean+=faces[mood][person]
        #print "mean descriptor length=",mood,np.linalg.norm(mean)
    mean/=7
    print "mean descriptor length=",np.linalg.norm(mean)
    meanPerson.append(mean)
#print meanPerson

print "Distance to mean "
k=1
for k in range(7):
    print "Mood ",k
    print k,
    for person0 in range(10):
        print "%03.2f" % person0,
    print
    for person0 in range(10):
        print person0,
        for person1 in range(10):
            d=distance(meanPerson[person0],faces[k][person1])
            print "%03.2f" % d,
        print
  
print "Distance to mood"
k=normal
for person0 in range(10):
    for person1 in range(10):
        d=distance(faces[k][person0],faces[k][person1])
        print "%3.2f" % d,
    print

person=0
for i in range(7):
    for j in range(7):
        d=distance(faces[i][person],faces[j][person])
        print "%3.2f" % d,
    print
    
# for i in range(70):
#     for j in range(70):
#         d=distance(faces[i/10][i%10],faces[j/10][j%10])
#         print "%3.2f" % d,
#     print     
#       
faces_des_all_idx=[(person,i,faces[i][person]) for i in range(7) for person in range(10) ]
#add mean face as 8th mood
for i in range(10):
    faces_des_all_idx.append((i,8,meanPerson[i]))
personId         =np.array([ d[0] for d in faces_des_all_idx])
moodId           =np.array([ d[1] for d in faces_des_all_idx])
faces_des_all    =np.array([ d[2] for d in faces_des_all_idx])
idxs=personId
#print idxs
#print faces_des_all
#X_embedded = TSNE(n_components=2,perplexity=5,method='exact').fit_transform(faces_des_all)
X_embedded = TSNE(n_components=2,perplexity=5).fit_transform(faces_des_all)
print "emb",X_embedded.shape

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple','red'
for i in range(max(idxs)):
    #idx=idxs==i
    if i>4:
        mk='s'
    else:
        mk='o'
    plt.scatter(X_embedded[idxs[:70]==i, 0], X_embedded[idxs[:70]==i, 1],marker=mk,c=colors[i],label=i)
#Mean faces are from 70 to 80
idxs[:70]=100
for i in range(10):
    plt.scatter(X_embedded[idxs==i, 0], X_embedded[idxs==i, 1],marker='D',c=colors[i],label=i)
plt.legend(bbox_to_anchor=(1, 1))


def dFaceNormal(person,mood):
    n=np.array(faces[normal][person])
    m=np.array(faces[mood][person])
    return n-m

def dFaceMean(person,mood):
    n=meanPerson[person]
    m=np.array(faces[mood][person])
    return n-m

#Lets play now with mean-mood difference vector
lDifMean=[]
for k in range(7):
    for person in range(10):
        lDifMean.append([person,k,dFaceMean(person,k)])
#Female vs all
print "Difference Female vs all"
print " ",
for person0 in range(10):
    print "%03.2f" % person0,
print
for k in range(7):
    print k,
    for person in range(10):
        d=distance(dFaceNormal(0,happy),dFaceNormal(person,k))
        print "%03.2f" % d,
    print    
  

dHappy0=dFaceMean(0,happy)
dShad0 =dFaceMean(0,shad)
dHappy1=dFaceMean(1,happy)
dShad1 =dFaceMean(1,shad)
print " %3.2f %3.2f" %(distance(dShad0,dShad1) ,distance(dHappy1,dHappy0))
print " %3.2f %3.2f" %(distance(dShad0,dHappy0),distance(dShad0,dHappy1))
print " %3.2f %3.2f" %(distance(dShad1,dHappy0),distance(dShad1,dHappy1))

#Male vs Female
print "Female and Male Mean faces descriptor for each person"
meanFemale=meanPerson[0].copy()
print "meanFemale descriptor length=",person,np.linalg.norm(meanFemale),np.linalg.norm(meanPerson[0])
for person in range(1,5):
    meanFemale+=meanPerson[person]
    print "meanFemale descriptor length=",person,np.linalg.norm(meanFemale),np.linalg.norm(meanPerson[person])
meanFemale/=5
print "meanFemale descriptor length=",np.linalg.norm(meanFemale)

meanMale=meanPerson[5].copy()
for person in range(6,10):
    meanMale+=meanPerson[person]
meanMale/=5
print "meanMale descriptor length=",np.linalg.norm(meanMale)

#Female vs all
print "Female vs all"
print " ",
for person0 in range(10):
    print "%03.2f" % person0,
print
for k in range(7):
    print k,
    for person in range(10):
        d=distance(meanFemale,faces[k][person])
        print "%03.2f" % d,
    print    

#Male vs all
print "Male vs all"
print " ",
for person0 in range(10):
    print "%03.2f" % person0,
print
for k in range(7):
    print k,
    for person in range(10):
        d=distance(meanMale,faces[k][person])
        print "%03.2f" % d,
    print    
    
#HappyFemale mean vs all 
#This doesn't seem to work but discriminates female from male
print "Happy mean face desriptor"
meanHappyFemale=faces[happy][0].copy()
for person in range(1,5):
    meanHappyFemale+=faces[happy][person]
    print "meanHappy descriptor length=",person,np.linalg.norm(meanHappyFemale),np.linalg.norm(meanPerson[person])
meanHappyFemale/=5
meanHappyMale=faces[happy][5].copy()
for person in range(6,10):
    meanHappyMale+=faces[happy][person]
    print "meanHappy descriptor length=",person,np.linalg.norm(meanHappyMale),np.linalg.norm(meanPerson[person])
meanHappyMale/=5
print "meanHappy descriptor length=",np.linalg.norm(meanHappyFemale)
print "HappyFemale vs female and HappyMale vs male"
print " ",
for person0 in range(10):
    print "%03.2f" % person0,
print
for k in range(7):
    print k,
    for person in range(5):
        d=distance(meanHappyFemale,faces[k][person])
        print "%03.2f" % d,
    for person in range(5,10):
        d=distance(meanHappyMale,faces[k][person])
        print "%03.2f" % d,
    print  


plt.show()
#dlib.hit_enter_to_continue()


