import os
import matplotlib.pyplot as plt
import numpy as np

valuesOutputDirectory = "OutputValues/"

listOfImageNames = os.listdir(valuesOutputDirectory)

areas=[]
perimeters=[]
# Read values from input files
for valueFile in os.listdir(valuesOutputDirectory):
    with open(valuesOutputDirectory+valueFile, 'r') as f:
        fullFile = f.read()[:-2] #remove last \n
        lines= fullFile.split("\n")
        for elements in lines:
            values=elements.split(";")
            areas.append(values[0])
            perimeters.append(values[1])


#Draw histograms

plt.subplot(121,title='Areas distribution')
areas=np.array(areas)
plt.hist(areas.astype(float),200,density=True)
plt.ticklabel_format(style='sci',scilimits=(-3,4),axis='both') #force scientific notation
plt.xlabel("Pixels")
plt.ylabel("Density")

plt.subplot(122,title='Perimeters distribution')
perimeters=np.array(perimeters)
plt.hist(perimeters.astype(float),200,density=True)
plt.ticklabel_format(style='sci',scilimits=(-3,4),axis='both') #force scientific notation
plt.xlabel("Pixels")
plt.ylabel("Density")
plt.show()