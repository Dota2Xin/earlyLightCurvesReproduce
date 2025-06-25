
import numpy as np
import bisect
from scipy.interpolate import interp1d
import os
import math


def readIntensities(intensityFile,numAngles, phiCount, thetaCount, radiusCount, radiusBlockSize, thetaBlockSize):
    intensities=np.zeros((phiCount,thetaCount,radiusCount, numAngles))
    with open(intensityFile, 'r') as file:
        for i, line in enumerate(file):
            phiIndex=int((i//(thetaBlockSize*radiusBlockSize)%phiCount))
            thetaIndex=int((i//radiusBlockSize)%thetaCount)
            radiusIndex=int(i%radiusBlockSize)
            currentVals=line.split()
            for k in range(len(currentVals)):
                intensities[phiIndex][thetaIndex][radiusIndex][k]=float(currentVals[k])
    return intensities

def readIntensitiesLocations(intensityFile,locationFile,numAngles,radii, phiCount, thetaCount, radiusCount):
    intensities=np.zeros((phiCount,thetaCount,radiusCount, numAngles))
    with open(intensityFile, 'r') as file, open(locationFile, 'r') as file2:
        locationLines=file2.readlines()
        for i, line in enumerate(file):
            currentLocations = locationLines[i].split()
            radius=float(currentLocations[0])
            theta = float(currentLocations[1])
            phi = float(currentLocations[2])
            thetaIndex = int(math.floor(300 * theta / np.pi))
            phiIndex = int(math.floor(30 * phi / (2 * np.pi)))
            radiusIndex=int(bisect.bisect_left(radii,radius)-1)
            currentVals=line.split()
            for k in range(len(currentVals)):
                intensities[phiIndex][thetaIndex][radiusIndex][k]=float(currentVals[k])
    return intensities

def constructDataArray(start, final, resolution, scaleFactor):
    if scaleFactor==1.0:
        return np.linspace(start, final, resolution)
    else:
        deltaR=final-start
        deltaR0=(1-scaleFactor)*deltaR/(1-scaleFactor**resolution)
        r=np.zeros(resolution)
        for i in range(resolution):
            r[i]=deltaR0*(1-scaleFactor**i)/(1-scaleFactor)
        return r+start

def interpolateVals(radius,theta,radiusResolution, data):
    newRadius=np.linspace(radius[0],radius[-1], radiusResolution)
    interpolatedData=np.zeros((len(theta), radiusResolution))

    for i in range(len(theta)):
        interp_func = interp1d(radius, data[i, :], kind='linear', fill_value="extrapolate")
        interpolatedData[i, :] = interp_func(newRadius)

    return interpolatedData

def interpolateIntensity(radius,theta,phi, radiusResolution,angleNum, data):
    newRadius = np.linspace(radius[0], radius[-1], radiusResolution)
    interpolatedData = np.zeros((len(phi),len(theta), radiusResolution, angleNum))

    for i in range(len(theta)):
        for j in range(angleNum):
            for k in range(len(phi)):
                interp_func = interp1d(radius, data[k,i, :, j], kind='linear', fill_value="extrapolate")
                interpolatedData[k,i, :, j] = interp_func(newRadius)

    return interpolatedData


def processFinalFiles(dataFolder, toFolder, resolution, verbose=False):
    outputPath = "restartFiles/" + str(toFolder)
    radiusPath=dataFolder+"/radius.bin"
    phiPath=dataFolder+"/phi.bin"
    thetaPath=dataFolder+"/theta.bin"
    intensityPath = outputPath + "/intensity.bin"

    files=[]
    intensityFiles=[]
    locationFiles=[]
    for filename in os.listdir(dataFolder):
        if filename.endswith(".bin"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)
        elif filename.startswith("intensitiesRestart"):
            filepath=os.path.join(dataFolder,filename)
            intensityFiles.append(filepath)
        elif filename.startswith("locationsRestart"):
            filepath=os.path.join(dataFolder,filename)
            locationFiles.append(filepath)


    radius = np.fromfile(radiusPath, dtype=np.float64)
    theta =np.fromfile(thetaPath, dtype=np.float64)
    phi =np.fromfile(phiPath, dtype=np.float64)

    newRadius = np.linspace(radius[0], radius[-1], resolution)

    intensities = np.asarray(readIntensitiesLocations(intensityFiles[0],locationFiles[0], 80,radius, len(phi), len(theta), len(radius)))
    interpolateIntensities = np.asarray(interpolateIntensity(radius,theta,phi,resolution, 80,intensities), dtype=np.float64)

    if not verbose:
        interpolateIntensities.tofile(intensityPath)


def processFinalFiles2(dataFolder, toFolder, resolution, verbose=False):
    outputPath = "restartFiles/" + str(toFolder)
    radiusPath=dataFolder+"/radius.bin"
    phiPath=dataFolder+"/phi.bin"
    thetaPath=dataFolder+"/theta.bin"
    intensityPath = outputPath + "/intensity.bin"

    files=[]
    intensityFiles=[]
    locationFiles=[]
    for filename in os.listdir(dataFolder):
        if filename.endswith(".bin"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)
        elif filename.startswith("intensitiesRestart"):
            filepath=os.path.join(dataFolder,filename)
            intensityFiles.append(filepath)
        elif filename.startswith("locationsRestart"):
            filepath=os.path.join(dataFolder,filename)
            locationFiles.append(filepath)


    radius = np.fromfile(radiusPath, dtype=np.float64)
    theta =np.fromfile(thetaPath, dtype=np.float64)
    phi =np.fromfile(phiPath, dtype=np.float64)
    print(len(radius))
    print(len(theta))
    print(len(phi))
    newRadius = np.linspace(radius[0], radius[-1], resolution)

    intensities = np.asarray(readIntensitiesLocations(intensityFiles[0],locationFiles[0], 80,radius, len(phi), len(theta), len(radius)))


    if not verbose:
        np.asarray(intensities, dtype=np.float64).tofile(intensityPath)





def main():
    #processFinalFiles("data/restartTestNew", "test",3000,verbose=False)
    processFinalFiles2("data/sixteenPercentRestart1", "sixteen1", 3000, verbose=False)

    return


if __name__=="__main__":
    main()







