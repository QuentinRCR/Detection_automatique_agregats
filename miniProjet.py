import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage as ski
import cv2

def applyCleaningFilter(image,filter):
    filteredImage = image.astype(int)-filter.astype(int) #go into int32 to avoid overflow issues
    # plt.hist(filteredImage.flatten(),255)
    # plt.show()
    # filteredImage[filteredImage>80]=80 #remove useless height values
    # filteredImage[filteredImage < -30] = -30
    # plt.hist(filteredImage.flatten(),256)
    # plt.show()
    mini = np.min(filteredImage)
    maxi = np.max(filteredImage)
    # return ski.exposure.equalize_hist(filteredImage,255)
    # return ((filteredImage - mini) / (maxi - mini) * 255).astype(np.uint8) #streatch image to be between 0 and 255
    return np.maximum(filteredImage,np.zeros(filteredImage.shape)).astype(np.uint8)

def generateFilter(listImages,imageShape):
    filter = np.zeros(imageShape, dtype=int)
    for imageName in listImages:
        image = ski.io.imread(imageName)[:,:,0]
        filter += image
    filter = filter // numberImages
    filter =filter.astype(np.uint8)
    if(not os.path.exists("filters")): #if the folder doesn't exist, create it
        os.mkdir("filters")
    ski.io.imsave("filters/luminosityFilter.bmp",filter,format='bmp')

imageDirectory = "Sous_ensemble_test"
listImages= [imageDirectory+"/"+nameImage for nameImage in os.listdir(imageDirectory)]
numberImages = len(listImages)
testImage= ski.io.imread(listImages[0])[:,:,0] #take only one component
imageShape = testImage.shape[0:2]

#if the luminosity filter in not yet saved, create it
if(not os.path.isfile("filters/luminosityFilter.bmp")):
    generateFilter(listImages,imageShape)

filters = ski.io.imread("filters/luminosityFilter.bmp") #get filter from saved images
plt.subplot(221,title="original image")
plt.imshow(testImage,"gray")

plt.subplot(222,title='cleaned equalized image')
cleanedEqualizedImage = ski.exposure.equalize_hist(applyCleaningFilter(testImage, filters))
plt.imshow(cleanedEqualizedImage, "gray")

#remove a bit of noise on the image
morphedImage = cv2.morphologyEx(cleanedEqualizedImage, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
plt.subplot(223,title="morphed open Image")
plt.imshow(morphedImage,"gray")

# uniformize blobs to avoid noise and bloobs fusions with filter
plt.subplot(224,title="blured image")
bluedImage = ski.filters.gaussian(morphedImage,1)
plt.imshow(bluedImage,"gray")

plt.figure()

#make the image binary
threshold=ski.filters.threshold_otsu(bluedImage)
binaryImage = bluedImage>threshold
plt.subplot(221,title="binary image")
binaryImage = binaryImage.astype(np.uint8) #transorm it to uint8
binaryImage[binaryImage==True]=255
plt.imshow(binaryImage,"gray")

#remove noise with median filter
medianImage = ski.filters.median(binaryImage,ski.morphology.disk(10) )
plt.subplot(222)
plt.imshow(medianImage,"gray")




'''''''''
#blure the image to avoid part of the noise
bluredImage = ski.filters.gaussian(cleanedImage,1)
bluredImage=ski.exposure.equalize_hist(bluredImage)

plt.subplot(223,title='cleaned blured equalized image')
plt.imshow(bluredImage,"gray")

# cleanedImage = cv2.morphologyEx(cleanedImage, cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))

plt.subplot(224,title='binary image')
threshold = ski.filters.threshold_local(cleanedImage,101) #101 is the size of the smallest things on the image
binaryImage = cleanedImage>threshold
binaryImage = binaryImage.astype(np.uint8)
binaryImage[binaryImage==True]=255
plt.imshow(binaryImage, "gray")

plt.figure()

# tranforme binary image
closedImage = cv2.dilate(binaryImage,np.ones((2,2),np.uint8),iterations = 1)

# closedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN,np.ones((1,1),np.uint8))
# closedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
# closedImage = cv2.morphologyEx(binaryImage, cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))

# plt.subplot(224,title="closed image")
plt.imshow(closedImage,"gray")
'''''''''
plt.show()
