import os
import matplotlib.pyplot as plt
import numpy as np

valuesOutputDirectory = "AreaAndPerimeterValues/"

listOfImageNames = os.listdir(valuesOutputDirectory)

areas=[]
perimeters=[]
for valueFile in os.listdir(valuesOutputDirectory):
    with open(valuesOutputDirectory+valueFile, 'r') as f:
        fullFile = f.read()[:-2] #remove last \n
        lines= fullFile.split("\n")
        for elements in lines:
            values=elements.split(";")
            areas.append(values[0])
            perimeters.append(values[1])

plt.subplot(121,title='areas')
areas=np.array(areas)
plt.hist(areas.astype(float),20)

plt.subplot(122,title='perimeters')
perimeters=np.array(perimeters)
plt.hist(perimeters.astype(float),20)
plt.show()