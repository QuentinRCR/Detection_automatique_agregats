import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage as ski
import cv2

def applyCleaningFilter(image,filter):
    filteredImage = image.astype(int)-filter.astype(int) #go into int32 to avoid overflow issues

    #remove every negative values created by the filter and replace them by 0
    return np.maximum(filteredImage,np.zeros(filteredImage.shape)).astype(np.uint8)

#generate the filter to remove the main noises. This filter is the sum of every images
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
testImage= ski.io.imread(listImages[1])[:,:,0] #take only one component
imageShape = testImage.shape[0:2]

#if the luminosity filter in not yet saved, create it
if(not os.path.isfile("filters/luminosityFilter.bmp")):
    generateFilter(listImages,imageShape)

filters = ski.io.imread("filters/luminosityFilter.bmp") #get filter from saved images

#print original image to show evolution
plt.subplot(221,title="original image")
plt.imshow(testImage,"gray")

#apply luminosity fiter and equalize image
cleanedEqualizedImage = ski.exposure.equalize_hist(applyCleaningFilter(testImage, filters))
plt.subplot(222,title='cleaned equalized image')
plt.imshow(cleanedEqualizedImage, "gray")

#remove a bit of noise on the image to have a darker background
morphedImage = cv2.morphologyEx(cleanedEqualizedImage, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
plt.subplot(223,title="darker background with morph open")
plt.imshow(morphedImage,"gray")

# uniformize blobs to avoid noise and bloobs fusions with filter
plt.subplot(224,title="blured image")
bluedImage = ski.filters.gaussian(morphedImage,1)
plt.imshow(bluedImage,"gray")

plt.figure()

#make the image binary with a local threshold
threshold=ski.filters.threshold_local(bluedImage,301)
binaryImage = bluedImage>threshold
plt.subplot(221,title="binary image")
binaryImage = binaryImage.astype(np.uint8) #transorm it to uint8
binaryImage[binaryImage==True]=255
plt.imshow(binaryImage,"gray")

#remove small noise
morphedBinaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
plt.subplot(222, title="remove noise")
plt.imshow(morphedBinaryImage,"gray")

#smooth image with median filter
medianImage = ski.filters.median(morphedBinaryImage,ski.morphology.disk(10) )
plt.subplot(223,title="smoothed image")
plt.imshow(medianImage,"gray")

#find contours, areas and blobs
contours,hierarchy = cv2.findContours(medianImage, 1, 2)
imageWithAreaAndPerimeter=cv2.cvtColor(medianImage, cv2.COLOR_GRAY2RGB)

#diplay area on the image
for i, cnt in enumerate(contours):
   M = cv2.moments(cnt)
   if M['m00'] != 0.0:
      x1 = int(M['m10']/M['m00'])
      y1 = int(M['m01']/M['m00'])
   area = cv2.contourArea(cnt)
   perimeter = cv2.arcLength(cnt, True)
   perimeter = round(perimeter, 4)
   print(f'Area of contour {i+1}:', area)
   print(f'Perimeter of contour {i+1}:', perimeter)
   imageWithAreaAndPerimeter = cv2.drawContours(imageWithAreaAndPerimeter, [cnt], -1, (0, 255, 255), 3)
   cv2.putText(imageWithAreaAndPerimeter, f'Area :{area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   cv2.putText(imageWithAreaAndPerimeter, f'Perimeter :{perimeter}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


plt.subplot(224,title='image with values')
plt.imshow(imageWithAreaAndPerimeter)

plt.figure()
# plt.title("final comparison")
plt.subplot(121,title="original image")
plt.imshow(testImage,"gray")

plt.subplot(122,title="threaded image")
plt.imshow(imageWithAreaAndPerimeter,"gray")

plt.show()