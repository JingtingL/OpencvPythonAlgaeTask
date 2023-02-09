import cv2
import numpy as np
print("Package Imported")

algae_list = ['algae1.jpg', 'algae2.jpg', 'algae3.jpg', 'algae4.jpg']

for i, img in enumerate(algae_list):
    algae = cv2.imread("AlgaePics/"+img)

    #gray scale------------------------------------
    algae_gray = cv2.cvtColor(algae, cv2.COLOR_BGR2GRAY) 
    cv2.imwrite("GrayScale/algae"+str(i+1)+"_gray.png",algae_gray)
    #binary------------------------------------
    ret,thresh = cv2.threshold(algae_gray,127,255,cv2.THRESH_BINARY)
    cv2.imwrite("Binary/algae"+str(i+1)+"_binary.png", thresh)
    #Adaptive Thresholding-----------------------------
    th = cv2.adaptiveThreshold(algae_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imwrite("AdaptiveThresholding/algae"+str(i+1)+"_at.png", th)
    #Adaptive Gaussian Thresholding------------------------------
    agt = cv2.adaptiveThreshold(algae_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imwrite("AdaptiveGausThres/algae"+str(i+1)+"_agt.png", agt)
    #erosion--------------------------------------------
    kernel = np.ones((5,5),np.uint8)
    e = cv2.erode(thresh,kernel,iterations = 1)
    cv2.imwrite("Erosion/algae"+str(i+1)+"_erosion.png", e)
    #Dilation-------------------------------------------------
    d = cv2.dilate(thresh,kernel,iterations = 1)
    cv2.imwrite("Dilation/algae"+str(i+1)+"_dilation.png", d)
    #Opening--------------------------------------------------
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("Opening/algae"+str(i+1)+"_opening.png", opening)
    #closing----------------------------------------------------
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("Closing/algae"+str(i+1)+"_closing.png", closing)
    #Morphological Gradient------------------------------------------------
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite("MorphGradient/algae"+str(i+1)+"_mg.png",gradient)
    #top hat ---------------------------------------------------------------
    tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite("TopHat/algae"+str(i+1)+"_th.png",tophat)
    #black hat ---------------------------------------------------------------
    blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite("BlackHat/algae"+str(i+1)+"_bh.png", blackhat)