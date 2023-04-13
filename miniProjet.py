import os;
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def applyCleaningFilter(image,filter):
    filteredImage = image.astype(float)-filter.astype(float) #go into float to avoid overflow issues
    mini = np.min(filteredImage)
    maxi = np.max(filteredImage)
    return ((filteredImage - mini) / (maxi - mini) * 255).astype(int) #streatch image to be between 0 and 255

imageDirectory = "3h300us"
listImages= [imageDirectory+"/"+nameImage for nameImage in os.listdir(imageDirectory)]
numberImages = len(listImages)
testImage= ski.io.imread(listImages[0])[:,:,0] #take only one component
imageShape = testImage.shape[0:2]


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

if(not os.path.isfile("filters/luminosityFilter.bmp")): #if the luminosity filter in not yet saved, create it
    generateFilter(listImages,imageShape)

filters = ski.io.imread("filters/luminosityFilter.bmp") #get filter from saved images
plt.subplot(221,title="filter")
plt.imshow(filters,"gray")
plt.subplot(222,title='cleaned image')
plt.imshow(applyCleaningFilter(testImage,filters),"gray")
plt.show()

