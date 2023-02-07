import cv2
import numpy as np
print("Package Imported")

algae1 = cv2.imread("AlgaePics/algae1.png")
algae2 = cv2.imread("AlgaePics/algae2.jpeg")
algae3 = cv2.imread("AlgaePics/algae3.jpeg")
algae4 = cv2.imread("AlgaePics/algae4.jpeg")

#gray scale------------------------------------
algae1_gray = cv2.cvtColor(algae1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayScale/algae1_gray.png",algae1_gray)

algae2_gray = cv2.cvtColor(algae2, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayScale/algae2_gray.jpeg",algae2_gray)
#cv2.imshow("algae2_gray", algae2_gray)

algae3_gray = cv2.cvtColor(algae3, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayScale/algae3_gray.jpeg",algae3_gray)
#cv2.imshow("algae3_gray", algae3_gray)

algae4_gray = cv2.cvtColor(algae4, cv2.COLOR_BGR2GRAY)
cv2.imwrite("GrayScale/algae4_gray.jpeg", algae4_gray)
#cv2.imshow("algae4_gray", algae4_gray)

#binary------------------------------------
ret,thresh1 = cv2.threshold(algae1_gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite("Binary/algae1_binary.png", thresh1)

ret,thresh2 = cv2.threshold(algae2_gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite("Binary/algae2_binary.jpeg", thresh2)

ret,thresh3 = cv2.threshold(algae3_gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite("Binary/algae3_binary.jpeg", thresh3)

ret,thresh4 = cv2.threshold(algae4_gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite("Binary/algae4_binary.jpeg", thresh4)

#Adaptive Thresholding-----------------------------
th1 = cv2.adaptiveThreshold(algae1_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveThresholding/algae1_at.png", th1)

th2 = cv2.adaptiveThreshold(algae2_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveThresholding/algae2_at.jpeg", th2)

th3 = cv2.adaptiveThreshold(algae3_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveThresholding/algae3_at.jpeg", th3)

th4 = cv2.adaptiveThreshold(algae4_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveThresholding/algae4_at.jpeg", th4)

#Adaptive Gaussian Thresholding------------------------------
agt1 = cv2.adaptiveThreshold(algae1_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveGausThres/algae1_at.png", agt1)

agt2 = cv2.adaptiveThreshold(algae2_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveGausThres/algae2_at.png", agt2)

agt3 = cv2.adaptiveThreshold(algae3_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveGausThres/algae3_at.png", agt3)

agt4 = cv2.adaptiveThreshold(algae4_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("AdaptiveGausThres/algae4_at.png", agt4)

#erosion----------------------------------
kernel = np.ones((5,5),np.uint8)
e1 = cv2.erode(thresh1,kernel,iterations = 1)
cv2.imwrite("Erosion/algae1_erosion.png", e1)

e2 = cv2.erode(thresh2,kernel,iterations = 1)
cv2.imwrite("Erosion/algae2_erosion.png", e2)

e3 = cv2.erode(thresh3,kernel,iterations = 1)
cv2.imwrite("Erosion/algae3_erosion.png", e3)

e4 = cv2.erode(thresh4,kernel,iterations = 1)
cv2.imwrite("Erosion/algae4_erosion.png", e4)

#Dilation-------------------------------------
d1 = cv2.dilate(thresh1,kernel,iterations = 1)
cv2.imwrite("Dilation/algae1_dilation.png", d1)

d2 = cv2.dilate(thresh2,kernel,iterations = 1)
cv2.imwrite("Dilation/algae2_dilation.png", d2)

d3 = cv2.dilate(thresh3,kernel,iterations = 1)
cv2.imwrite("Dilation/algae3_dilation.png", d3)

d4 = cv2.dilate(thresh4,kernel,iterations = 1)
cv2.imwrite("Dilation/algae4_dilation.png", d4)

#Opening--------------------------------------------------
opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
cv2.imwrite("Opening/algae1_opening.png", opening1)

opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
cv2.imwrite("Opening/algae2_opening.png", opening2)

opening3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
cv2.imwrite("Opening/algae3_opening.png", opening3)

opening4 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
cv2.imwrite("Opening/algae4_opening.png", opening4)

#closing----------------------------------------------------
closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("Closing/algae1_closing.png", closing1)

closing2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("Closing/algae2_closing.jpeg", closing2)

closing3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("Closing/algae3_closing.jpeg", closing3)

closing4 = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("Closing/algae4_closing.jpeg", closing4)

#Morphological Gradient------------------------------------------------
gradient1 = cv2.morphologyEx(thresh1, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite("MorphGradient/algae1_mg.png",gradient1)

gradient2 = cv2.morphologyEx(thresh2, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite("MorphGradient/algae2_mg.png",gradient2)

gradient3 = cv2.morphologyEx(thresh3, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite("MorphGradient/algae3_mg.png",gradient3)

gradient4 = cv2.morphologyEx(thresh4, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite("MorphGradient/algae4_mg.png",gradient4)

#top hat ---------------------------------------------------------------
tophat1 = cv2.morphologyEx(thresh1, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite("TopHat/algae1_th.png",tophat1)

tophat2 = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite("TopHat/algae2_th.png",tophat2)

tophat3 = cv2.morphologyEx(thresh3, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite("TopHat/algae3_th.png",tophat3)

tophat4 = cv2.morphologyEx(thresh4, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite("TopHat/algae4_th.png",tophat4)

#black hat
blackhat1 = cv2.morphologyEx(thresh1, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite("BlackHat/algae1_bh.png", blackhat1)

blackhat2 = cv2.morphologyEx(thresh2, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite("BlackHat/algae2_bh.png", blackhat2)

blackhat3 = cv2.morphologyEx(thresh3, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite("BlackHat/algae3_bh.png", blackhat3)

blackhat4 = cv2.morphologyEx(thresh4, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite("BlackHat/algae4_bh.png", blackhat4)

cv2.waitKey(1)
