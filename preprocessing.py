import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import cv2

imageInputDirectory = "InputImages/"
imageOutputDirectory = "OutputImages/"
valuesOutputDirectory = "OutputValues/"

def applyCleaningFilter(image,filter):
    filteredImage = image.astype(int)-filter.astype(int) #go into int32 to avoid overflow issues

    #remove every negative values created by the filter and replace them by 0
    return np.maximum(filteredImage,np.zeros(filteredImage.shape)).astype(np.uint8)


#generate the filter to remove the main noises. This filter is the sum of every images
def generateFilter(inputDirectory,listImages,imageShape):
    filter = np.zeros(imageShape, dtype=int)
    for imageName in listImages:
        image = ski.io.imread(inputDirectory+imageName)[:,:,0]
        filter += image
    filter = filter // len(listImages)
    filter =filter.astype(np.uint8)
    if(not os.path.exists("filters")): #if the folder doesn't exist, create it
        os.mkdir("filters")
    ski.io.imsave("filters/luminosityFilter.bmp",filter,format='bmp')


def removeElementsTouchingEdge(image):
    # add 1 pixel white border all around
    pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = pad.shape

    # create zeros mask 2 pixels larger in each dimension
    mask = np.zeros([h + 2, w + 2], np.uint8)

    # floodfill outer white border with black
    img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]

    return img_floodfill


def getAreaAndPerimeters(image):
    areas=[]
    perimeters=[]

    #get all the contours
    contours, hierarchy = cv2.findContours(image, 1, 2)
    imageWithAreaAndPerimeter = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # display and perimeter on the image
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            x1 = int(M['m10'] / M['m00'])
            y1 = int(M['m01'] / M['m00'])
        area = cv2.contourArea(cnt)
        areas.append(area)
        perimeter = cv2.arcLength(cnt, True)
        perimeters.append(perimeter)
        perimeter = round(perimeter, 4)
        imageWithAreaAndPerimeter = cv2.drawContours(imageWithAreaAndPerimeter, [cnt], -1, (0, 255, 255), 3)
        cv2.putText(imageWithAreaAndPerimeter, f'Area :{area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(imageWithAreaAndPerimeter, f'Perimeter :{perimeter}', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
    return (areas,perimeters,imageWithAreaAndPerimeter)

#==========Beginning of the script=========================

# Get the name of every image in the inputDirectory
listOfImageNames = os.listdir(imageInputDirectory)
imageShape = ski.io.imread(imageInputDirectory+listOfImageNames[0]).shape[0:2]

#For every image process it
for imageName in listOfImageNames:

    # it the image was already processed, skip it
    if(os.path.isfile(valuesOutputDirectory+imageName[:-4]+".csv")):
        continue


    print("Processing ",imageName)
    processedImage= ski.io.imread(imageInputDirectory + imageName)[:, :, 0] #take only one component

    #if the luminosity filter in not yet saved, create it
    if(not os.path.isfile("filters/luminosityFilter.bmp")):
        generateFilter(imageInputDirectory,listOfImageNames,imageShape)

    filters = ski.io.imread("filters/luminosityFilter.bmp") #get filter from saved images

    #print original image to show evolution
    plt.subplot(221,title="Original image")
    plt.imshow(processedImage, "gray")
    plt.axis("off")


    #apply luminosity fiter and equalize image
    cleanedEqualizedImage = ski.exposure.equalize_hist(applyCleaningFilter(processedImage, filters))
    plt.subplot(222,title='Filtered equalized image')
    plt.imshow(cleanedEqualizedImage, "gray")
    plt.axis("off")

    #remove a bit of noise on the image to have a darker background
    morphedImage = cv2.morphologyEx(cleanedEqualizedImage, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    plt.subplot(223,title="Darker background with morph open")
    plt.imshow(morphedImage,"gray")
    plt.axis("off")

    # uniformize aggregates to avoid noise and aggregates when applying the segmentation
    plt.subplot(224,title="Darker background with Morph Open")
    bluedImage = ski.filters.gaussian(morphedImage,1)
    plt.imshow(bluedImage,"gray")
    plt.axis("off")

    plt.figure()

    #make the image binary with a local threshold
    threshold=ski.filters.threshold_local(bluedImage,301)
    binaryImage = bluedImage>threshold
    plt.subplot(221,title="Binary image")
    binaryImage = binaryImage.astype(np.uint8) #transorm it to uint8 to be a valid cv2 input
    binaryImage[binaryImage==True]=255
    plt.imshow(binaryImage,"gray")
    plt.axis("off")

    #remove single pixel noises from the background
    morphedBinaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    plt.subplot(222, title="Morphed binary image")
    plt.imshow(morphedBinaryImage,"gray")
    plt.axis("off")

    #smooth image with median filter
    medianImage = ski.filters.median(morphedBinaryImage,ski.morphology.disk(10) )
    plt.subplot(223,title="Smoothed binary image")
    plt.imshow(medianImage,"gray")
    plt.axis("off")

    #remove elements touching the edges
    cleanEdgeImage =removeElementsTouchingEdge(medianImage)
    plt.subplot(224,title="Remove edge elements image")
    plt.axis("off")
    plt.imshow(cleanEdgeImage,"gray")


    #Find each aggregate, get their perimeters and areas and  display the info on the image
    areas,perimeters,imageWithAreaAndPerimeter=getAreaAndPerimeters(cleanEdgeImage)

    # saved the processed image and its corresponding values in a texte file
    plt.imsave(imageOutputDirectory+imageName,imageWithAreaAndPerimeter)
    with open(valuesOutputDirectory+imageName[:-4]+".csv", 'w') as f:
        for area,perimeter in zip(areas,perimeters):
            f.write(str(area)+";"+str(perimeter)+"\n")

    plt.figure()

    # Display a final comparison
    plt.subplot(121,title="Original image")
    plt.imshow(processedImage, "gray")
    plt.axis("off")

    plt.subplot(122,title="Final image")
    plt.imshow(imageWithAreaAndPerimeter,"gray")
    plt.axis("off")

    # plt.show()