import ast
import numpy as np
import athena_read
import h5py
import bisect
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import os
import math


def processFiles(files, folderName):
    newPath="C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/"+str(folderName)
    notOpened=True
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    else:
        notOpened=False
    densityFilePath=newPath+"/density.txt"
    pressureFilePath=newPath+"/pressure.txt"
    velocityFilePath=newPath+"/radialVelocity.txt"
    paramsFile=newPath+"/params.txt"
    timeFile=newPath+"/times.txt"
    filePaths=[densityFilePath,pressureFilePath,velocityFilePath,timeFile]
    densityData=[]
    pressureData=[]
    velocityData=[]
    paramsData=[]
    timeData=[]
    '''
    Fill our params in format: 1st Line=Radius Data, 2nd Line=Edge index we look at, 3rd=Theta Data, 4th=Theta Edge
    5th=initial Time of simulation if supernova is at t=0
    '''
    dataFile=athena_read.athdf(files[0])
    paramsData.append(dataFile['x1f'])
    paramsData.append(len(dataFile['x1f'])-2)
    paramsData.append(dataFile['x2f'])
    paramsData.append(len(dataFile['x2f']) - 2)
    vMax=2.53 * (10 ** 9)
    paramsData.append(dataFile['x1f'][0] /vMax)
    for i in range(len(files)):
        dataFile=athena_read.athdf(files[i])
        timeData.append(i)
        finalIndex=len(dataFile['x1f'])-2
        finalTheta=len(dataFile['x2f']) - 2
        densityStore=[]
        pressureStore=[]
        velocityStore=[]

        for j in range(finalTheta):
            densityStore.append(dataFile['rho'][0][j][finalIndex])
            pressureStore.append(dataFile['press'][0][j][finalIndex])
            velocityStore.append(dataFile['vel1'][0][j][finalIndex])
        densityData.append(densityStore)
        pressureData.append(pressureStore)
        velocityData.append(velocityStore)
    dataStores=[densityData,pressureData,velocityData,timeData]
    #Write to params first
    if notOpened:
        file=open(paramsFile,'w')
        for line in paramsData:
            file.write(str(line)+"\n")
        for i in range(len(filePaths)):
            file=open(filePaths[i],'w')
            for j in dataStores[i]:
                file.write(str(j)+"\n")
    else:
        for i in range(len(filePaths)):
            file=open(filePaths[i],'a')
            for j in dataStores[i]:
                file.write(str(j)+"\n")

def getDataDict(folderName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(folderName)
    densityFilePath = newPath + "/density.txt"
    pressureFilePath = newPath + "/pressure.txt"
    velocityFilePath = newPath + "/radialVelocity.txt"
    densityData=[]
    pressureData=[]
    velocityData=[]
    with open(densityFilePath, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            densityData.append(data)

    with open(pressureFilePath, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            pressureData.append(data)

    with open(velocityFilePath, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            velocityData.append(data)

    dataDict={"density": densityData, "pressure": pressureData, "velocity": velocityData}

    return dataDict

def lowerFloor(directoryFrom, directoryTo, floorFactor):
    oldPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(directoryFrom)
    newPath= "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(directoryTo)
    densityFilePathOld = oldPath + "/rho.txt"
    pressureFilePathOld = oldPath + "/pres.txt"
    velocityFilePathOld = oldPath + "/vr.txt"
    densityFilePathNew = newPath + "/rho.txt"
    pressureFilePathNew = newPath + "/pres.txt"
    densityData = []
    pressureData = []
    velocityData=[]

    with open(velocityFilePathOld, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            velocityData.append(data)

    with open(densityFilePathOld, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            densityData.append(data)

    with open(densityFilePathNew, 'w') as file:
        for i in range(len(densityData)):
            if velocityData[i]==0.0:
                file.write(str(densityData[i]/floorFactor)+"\n")
            else:
                file.write(str(densityData[i])+"\n")

    with open(pressureFilePathOld, 'r') as file:
        for line in file:
            lineString=line.strip()
            data=ast.literal_eval(lineString)
            pressureData.append(data)

    with open(pressureFilePathNew, 'w') as file:
        for i in range(len(pressureData)):
            if velocityData[i]==0.0:
                file.write(str(pressureData[i]/floorFactor)+"\n")
            else:
                file.write(str(pressureData[i])+"\n")

def createDataGrid(directoryFrom, directoryTo, resolutionR, resolutionTheta, verbose=False):
    oldPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(directoryFrom)
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(directoryTo)
    densityFilePathOld = oldPath + "/rho.txt"
    pressureFilePathOld = oldPath + "/pres.txt"
    velocityFilePathOld = oldPath + "/vr.txt"
    thetaFilePathOld = oldPath + "/theta.txt"
    radiusFilePathOld = oldPath + "/r.txt"

    densityOld=[]
    pressureOld=[]
    velocityOld=[]
    thetaOld=[]
    radiusOld=[]
    filePaths=[densityFilePathOld, pressureFilePathOld, velocityFilePathOld, thetaFilePathOld, radiusFilePathOld]
    storeArrays = [densityOld, pressureOld, velocityOld, thetaOld, radiusOld]

    for i in range(len(filePaths)):
        with open(filePaths[i], 'r') as file:
            for line in file:
                lineString=line.strip()
                data=ast.literal_eval(lineString)
                storeArrays[i].append(data)

    actualRadius=storeArrays[-1][0:1000]
    actualTheta=storeArrays[-2][0:-1:1000]
    actualTheta=actualTheta[0:-1]
    minRadius=np.min(storeArrays[-1])
    maxRadius=np.max(storeArrays[-1])

    thetaMin=0
    thetaMax=np.pi

    radiusGrid=np.linspace(minRadius+.00001,maxRadius-.00001, resolutionR)
    thetaGrid=np.linspace(thetaMin+.00001, thetaMax-.00001, resolutionTheta)

    density=[]
    pressure=[]
    velocity=[]
    theta=[]
    radius=[]

    #print(actualTheta)
    for i in range(len(radiusGrid)):
        radiusVal=radiusGrid[i]
        radiusIndex=bisect.bisect_left(actualRadius, radiusVal)-1
        density.append([])
        pressure.append([])
        velocity.append([])
        for j in range(len(thetaGrid)):
            thetaVal=thetaGrid[j]
            if thetaVal>actualTheta[-1]:
                thetaIndex=bisect.bisect_left(actualTheta, actualTheta[-1]-.00001)-1
            else:
                thetaIndex=bisect.bisect_left(actualTheta, thetaVal)-1
            totalIndex=thetaIndex*1000+radiusIndex
            density[i].append(storeArrays[0][totalIndex])
            pressure[i].append(storeArrays[1][totalIndex])
            velocity[i].append(storeArrays[2][totalIndex])

    storeArrays2D = [density, pressure, velocity, thetaGrid, radiusGrid]
    maxVel=np.zeros(len(thetaGrid))
    for i in range(len(thetaGrid)):
        for j in range(len(radiusGrid)-4):
            if density[j+5][i]/density[j+4][i]<1e-5:
                maxVel[i]=radiusGrid[j+5]/8200
                break
    if verbose:
        plt.polar()
        plt.title(f"Sound Speed over space")
        plt.imshow(np.asarray(density)[:-1,:-1], aspect='auto', cmap='inferno', origin='lower',extent=(thetaGrid[0], thetaGrid[-1], radiusGrid[0], radiusGrid[-1]), norm=LogNorm(vmin=1e-10,vmax=1e-6))
        #im = plt.pcolormesh(thetaGrid, radiusGrid, (5.0/3.0*(np.asarray(pressure)[:-1,:-1]/np.asarray(density)[:-1,:-1]))**.5, cmap='inferno', norm=LogNorm(vmin=1e2,vmax=1e8)) # , vmin=0,vmax=1)
        plt.colorbar(label="g/cm^3")
        plt.show()
    print(radiusGrid[0]/8200)
    print(radiusGrid[-1]/8200)
    return storeArrays2D, maxVel

def makeBinaryFile(storeArrays2D, binaryDest):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/binaryInputData/" + str(binaryDest)
    densityPath=newPath+"/density.bin"
    pressurePath = newPath + "/pressure.bin"
    params=newPath+"/params.txt"
    radiusLength=len(storeArrays2D[-1])
    radiusMin=storeArrays2D[-1][0]
    radiusMax=storeArrays2D[-1][-1]
    thetaLength=len(storeArrays2D[-2])
    thetaMin = storeArrays2D[-2][0]
    thetaMax = storeArrays2D[-2][-1]
    time=8200
    paramsData=[radiusLength, radiusMin, radiusMax, thetaLength, thetaMin, thetaMax, time]
    notOpened = True
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    else:
        notOpened = False

    file=open(params,'w')
    for line in paramsData:
        file.write(str(line)+"\n")

    density=np.asarray(storeArrays2D[0], dtype=np.float64)
    density.tofile(densityPath)
    pressure=np.asarray(storeArrays2D[1],dtype=np.float64)
    pressure.tofile(pressurePath)

def makeBinaryFileVel(storeArrays2D, maxVelocities, binaryDest):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/binaryInputData/" + str(binaryDest)
    densityPath=newPath+"/density.bin"
    pressurePath = newPath + "/pressure.bin"
    maxVelPath = newPath + "/maxVel.bin"
    params=newPath+"/params.txt"
    radiusLength=len(storeArrays2D[-1])
    radiusMin=storeArrays2D[-1][0]
    radiusMax=storeArrays2D[-1][-1]
    thetaLength=len(storeArrays2D[-2])
    thetaMin = storeArrays2D[-2][0]
    thetaMax = storeArrays2D[-2][-1]
    time=8200
    paramsData=[radiusLength, radiusMin/time, radiusMax/time, thetaLength, thetaMin, thetaMax, time]
    notOpened = True
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    else:
        notOpened = False

    file=open(params,'w')
    for line in paramsData:
        file.write(str(line)+"\n")

    density=np.asarray(storeArrays2D[0], dtype=np.float64)
    density.tofile(densityPath)
    pressure=np.asarray(storeArrays2D[1],dtype=np.float64)
    pressure.tofile(pressurePath)
    maxVel=np.asarray(maxVelocities, dtype=np.float64)
    maxVel.tofile(maxVelPath)

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
    newRadius=np.linspace(radius[0]+.001,radius[-1]-.001, radiusResolution)
    interpolatedData=np.zeros((len(theta), radiusResolution))

    for i in range(len(theta)):
        interp_func = interp1d(radius, data[i, :], kind='linear', fill_value="extrapolate")
        interpolatedData[i, :] = interp_func(newRadius)

    return interpolatedData
def makeAngles(directory):
    # Define the naming scheme of the files
    prefix = "fullAngles"
    extension = ".txt"

    # Create an empty list to store the data from all files
    files = []

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(extension) and filename.startswith(prefix):
            filepath = os.path.join(directory, filename)
            files.append(filepath)

    angles=files[0]
    normalVectors=np.zeros((80))
    with open(angles,'r') as file:
        for i,line in enumerate(file):
            currentVals=line.split()
            normalVectors[i]=currentVals[4]
    return normalVectors

def calculateRadiationEnergy(intensityPoint, angleWeights):
    #calculate radiation energy using yan-fei's W nephew scheme
    return 4*np.pi*np.sum(intensityPoint*angleWeights)

def makeRadEnergySlice(interpolatedIntensities, angleWeights, thetaSize, radialSize):
    radEnergy=np.zeros((thetaSize, radialSize))
    for i in range(thetaSize):
        for j in range(radialSize):
            radEnergy[i][j]=calculateRadiationEnergy(interpolatedIntensities[0][i][j], angleWeights)
    return radEnergy

def interpolateIntensity(radius,theta,phi, radiusResolution,angleNum, data):
    newRadius = np.linspace(radius[0]+.001, radius[-1]-0.001, radiusResolution)
    interpolatedData = np.zeros((len(phi),len(theta), radiusResolution, angleNum))

    for i in range(len(theta)):
        for j in range(angleNum):
            for k in range(len(phi)):
                interp_func = interp1d(radius, data[k,i, :, j], kind='linear', fill_value="extrapolate")
                interpolatedData[k,i, :, j] = interp_func(newRadius)

    return interpolatedData
def plotRadiationEnergy(dataFolder, toFolder, resolution):
    outputPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/restartFiles/" + str(toFolder)
    densityPath = outputPath + "/density.bin"
    pressurePath = outputPath + "/pressure.bin"
    radialVelPath = outputPath + "/radialVel.bin"
    intensityPath = outputPath + "/intensity.bin"
    paramsPath = outputPath + "/params.txt"
    angleWeights = makeAngles(dataFolder)
    files = []
    intensityFiles = []
    locationFiles = []
    for filename in os.listdir(dataFolder):
        if filename.endswith(".athdf"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)
        elif filename.startswith("intensitiesRestart"):
            filepath = os.path.join(dataFolder, filename)
            intensityFiles.append(filepath)
        elif filename.startswith("locationsRestart"):
            filepath = os.path.join(dataFolder, filename)
            locationFiles.append(filepath)
    finalFile = files[-1]
    data = athena_read.athdf(finalFile)

    radius = constructDataArray(data['RootGridX1'][0], data['RootGridX1'][1], data['RootGridSize'][0],
                                data['RootGridX1'][2])
    theta = constructDataArray(data['RootGridX2'][0], data['RootGridX2'][1], data['RootGridSize'][1],
                               data['RootGridX2'][2])
    phi = constructDataArray(data['RootGridX3'][0], data['RootGridX3'][1], data['RootGridSize'][2],
                             data['RootGridX3'][2])
    print(radius[-1])
    intensities=np.fromfile(intensityPath,dtype=np.float64)
    intensities=intensities.reshape(len(phi),len(theta),resolution, 80)
    plt.polar()
    newRadius = np.linspace(radius[0], radius[-1], resolution)
    radEnergy = makeRadEnergySlice(intensities, angleWeights, len(theta), resolution)
    plt.title(f"Density over space")
    # plt.imshow(np.asarray(interpolateDensity), aspect='auto', cmap='inferno', origin='lower',extent=(theta[0], theta[-1], newRadius[0], newRadius[-1]),norm=LogNorm(vmin=1e-10, vmax=1e-6))
    im = plt.pcolormesh(theta, newRadius, np.transpose(np.asarray(radEnergy[:-1, :-1])), cmap='inferno',
                        norm=LogNorm(vmin=1e-6, vmax=1e6))  # , vmin=0,vmax=1)
    # im = plt.pcolormesh(theta, radius, np.transpose(np.asarray(density)), cmap='inferno',norm=LogNorm(vmin=1e-13, vmax=1e-7))  # , vmin=0,vmax=1)

    plt.colorbar(label="g/cm^3")
    plt.show()

def plotRadiationEnergyBinary(dataFolder, toFolder, resolution):
    outputPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/restartFiles/" + str(toFolder)
    densityPath = outputPath + "/density.bin"
    pressurePath = outputPath + "/pressure.bin"
    radialVelPath = outputPath + "/radialVel.bin"
    intensityPath = outputPath + "/intensity.bin"
    paramsPath = outputPath + "/params.txt"
    angleWeights = makeAngles(dataFolder)
    files = []
    intensityFiles = []
    locationFiles = []
    for filename in os.listdir(dataFolder):
        if filename.endswith(".athdf"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)
        elif filename.startswith("intensitiesRestart"):
            filepath = os.path.join(dataFolder, filename)
            intensityFiles.append(filepath)
        elif filename.startswith("locationsRestart"):
            filepath = os.path.join(dataFolder, filename)
            locationFiles.append(filepath)
    finalFile = files[-1]
    data = athena_read.athdf(finalFile)

    radius = constructDataArray(data['RootGridX1'][0], data['RootGridX1'][1], data['RootGridSize'][0],
                                data['RootGridX1'][2])
    theta = constructDataArray(data['RootGridX2'][0], data['RootGridX2'][1], data['RootGridSize'][1],
                               data['RootGridX2'][2])
    phi = constructDataArray(data['RootGridX3'][0], data['RootGridX3'][1], data['RootGridSize'][2],
                             data['RootGridX3'][2])
    print(radius[-1])
    intensities=np.fromfile(intensityPath,dtype=np.float64)
    intensities=intensities.reshape(len(phi),len(theta),resolution, 80)
    plt.polar()
    newRadius = np.linspace(radius[0], radius[-1], resolution)
    radEnergy = makeRadEnergySlice(intensities, angleWeights, len(theta), resolution)
    plt.title(f"Density over space")
    # plt.imshow(np.asarray(interpolateDensity), aspect='auto', cmap='inferno', origin='lower',extent=(theta[0], theta[-1], newRadius[0], newRadius[-1]),norm=LogNorm(vmin=1e-10, vmax=1e-6))
    im = plt.pcolormesh(theta, newRadius, np.transpose(np.asarray(radEnergy[:-1, :-1])), cmap='inferno',
                        norm=LogNorm(vmin=1e-6, vmax=1e6))  # , vmin=0,vmax=1)
    # im = plt.pcolormesh(theta, radius, np.transpose(np.asarray(density)), cmap='inferno',norm=LogNorm(vmin=1e-13, vmax=1e-7))  # , vmin=0,vmax=1)

    plt.colorbar(label="g/cm^3")
    plt.show()

def processFinalFiles(dataFolder, toFolder, resolution, verbose=False):
    outputPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/restartFiles/" + str(toFolder)
    densityPath=outputPath+"/density.bin"
    pressurePath = outputPath + "/pressure.bin"
    radialVelPath = outputPath + "/radialVel.bin"
    intensityPath = outputPath + "/intensity.bin"
    paramsPath=outputPath+"/params.txt"
    #angleWeights=makeAngles(dataFolder)
    files=[]
    intensityFiles=[]
    locationFiles=[]
    for filename in os.listdir(dataFolder):
        if filename.endswith(".athdf"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)
        elif filename.startswith("intensitiesRestart"):
            filepath=os.path.join(dataFolder,filename)
            intensityFiles.append(filepath)
        elif filename.startswith("locationsRestart"):
            filepath=os.path.join(dataFolder,filename)
            locationFiles.append(filepath)
    finalFile=files[-1]
    data=athena_read.athdf(finalFile)

    radius = constructDataArray(data['RootGridX1'][0], data['RootGridX1'][1], data['RootGridSize'][0],data['RootGridX1'][2])
    theta = constructDataArray(data['RootGridX2'][0], data['RootGridX2'][1], data['RootGridSize'][1],data['RootGridX2'][2])
    phi = constructDataArray(data['RootGridX3'][0], data['RootGridX3'][1], data['RootGridSize'][2],data['RootGridX3'][2])
    print(radius[-1])

    density=data['rho'][0]
    pressure=data['press'][0]
    radialVel=data['vel1'][0]
    newRadius = np.linspace(radius[0]+.001, radius[-1]-.001, resolution)
    print(newRadius[-1])
    print(newRadius[0])
    print("1")
    interpolateDensity=np.asarray(interpolateVals(radius,theta,resolution, density), dtype=np.float64)
    print("2")
    interpolatePressure = np.asarray(interpolateVals(radius,theta,resolution, pressure), dtype=np.float64)
    print("3")
    interpolateRadialVel = np.asarray(interpolateVals(radius,theta, resolution,radialVel), dtype=np.float64)
    #print("4")
    #print(intensityFiles[0])
    #print(locationFiles[0])
    #intensities = np.asarray(readIntensitiesLocations(intensityFiles[0],locationFiles[0], 80,radius, len(phi), len(theta), len(radius)))
    #interpolateIntensities = np.asarray(interpolateIntensity(radius,theta,phi,resolution, 80,intensities), dtype=np.float64)

    if not verbose:
        interpolateDensity.tofile(densityPath)
        interpolatePressure.tofile(pressurePath)
        interpolateRadialVel.tofile(radialVelPath)
        interpolateIntensities.tofile(intensityPath)

        with open(paramsPath, 'w') as file:
            file.write("Initial Radius:"+str(radius[0])+"\n")
            file.write("Final Radius:"+str(radius[-1])+"\n")
            file.write("Resolution:"+str(resolution)+"\n")
    else:
        plt.polar()
        #radEnergy=makeRadEnergySlice(interpolateIntensities, angleWeights,len(theta), resolution)
        plt.title(f"Density over space")
        #plt.imshow(np.asarray(interpolateDensity), aspect='auto', cmap='inferno', origin='lower',extent=(theta[0], theta[-1], newRadius[0], newRadius[-1]),norm=LogNorm(vmin=1e-10, vmax=1e-6))
        im = plt.pcolormesh(theta, newRadius,np.transpose(np.asarray(interpolateDensity[:-1,:-1])), cmap='inferno', norm=LogNorm(vmin=1e-12,vmax=1e-6)) # , vmin=0,vmax=1)
        #im = plt.pcolormesh(theta, radius, np.transpose(np.asarray(density)), cmap='inferno',norm=LogNorm(vmin=1e-13, vmax=1e-7))  # , vmin=0,vmax=1)

        plt.colorbar(label="g/cm^3")
        plt.show()

def processFinalFilesAthdf(dataFolder, toFolder, resolution, verbose=False):
    outputPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/restartFiles/" + str(toFolder)
    densityPath=outputPath+"/density.bin"
    pressurePath = outputPath + "/pressure.bin"
    radialVelPath = outputPath + "/radialVel.bin"
    radiusPath=outputPath+"/radius.bin"
    phiPath=outputPath+"/phi.bin"
    thetaPath=outputPath+"/theta.bin"
    paramsPath=outputPath+"/params.txt"
    files=[]
    for filename in os.listdir(dataFolder):
        if filename.endswith(".athdf"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)

    finalFile=files[-1]
    data=athena_read.athdf(finalFile)

    radius =np.asarray(constructDataArray(data['RootGridX1'][0], data['RootGridX1'][1], data['RootGridSize'][0],data['RootGridX1'][2]),dtype=np.float64)
    theta = np.asarray(constructDataArray(data['RootGridX2'][0], data['RootGridX2'][1], data['RootGridSize'][1],data['RootGridX2'][2]),dtype=np.float64)
    phi = np.asarray(constructDataArray(data['RootGridX3'][0], data['RootGridX3'][1], data['RootGridSize'][2],data['RootGridX3'][2]),dtype=np.float64)
    print(radius[-1])

    density=data['rho'][0]
    pressure=data['press'][0]
    radialVel=data['vel1'][0]
    newRadius = np.linspace(radius[0], radius[-1], resolution)
    print("1")
    interpolateDensity=np.asarray(interpolateVals(radius,theta,resolution, density), dtype=np.float64)
    print("2")
    interpolatePressure = np.asarray(interpolateVals(radius,theta,resolution, pressure), dtype=np.float64)
    print("3")
    interpolateRadialVel = np.asarray(interpolateVals(radius,theta, resolution,radialVel), dtype=np.float64)
    print("4")

    if not verbose:
        interpolateDensity.tofile(densityPath)
        interpolatePressure.tofile(pressurePath)
        interpolateRadialVel.tofile(radialVelPath)
        radius.tofile(radiusPath)
        phi.tofile(phiPath)
        theta.tofile(thetaPath)

        with open(paramsPath, 'w') as file:
            file.write("Initial Radius:"+str(radius[0])+"\n")
            file.write("Final Radius:"+str(radius[-1])+"\n")
            file.write("Resolution:"+str(resolution)+"\n")
    else:
        plt.polar()
        plt.title(f"Density over space")
        #plt.imshow(np.asarray(interpolateDensity), aspect='auto', cmap='inferno', origin='lower',extent=(theta[0], theta[-1], newRadius[0], newRadius[-1]),norm=LogNorm(vmin=1e-10, vmax=1e-6))
        im = plt.pcolormesh(theta, newRadius,np.transpose(np.asarray(interpolateDensity[:-1,:-1])), cmap='inferno', norm=LogNorm(vmin=1e-12,vmax=1e-4)) # , vmin=0,vmax=1)
        #im = plt.pcolormesh(theta, radius, np.transpose(np.asarray(density)), cmap='inferno',norm=LogNorm(vmin=1e-13, vmax=1e-7))  # , vmin=0,vmax=1)

        plt.colorbar(label="g/cm^3")
        plt.show()

def processFinalFilesAthdf2(dataFolder, toFolder, resolution, verbose=False):
    outputPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/restartFiles/" + str(toFolder)
    densityPath=outputPath+"/density.bin"
    pressurePath = outputPath + "/pressure.bin"
    radialVelPath = outputPath + "/radialVel.bin"
    radiusPath=outputPath+"/radius.bin"
    phiPath=outputPath+"/phi.bin"
    thetaPath=outputPath+"/theta.bin"
    paramsPath=outputPath+"/params.txt"
    files=[]
    for filename in os.listdir(dataFolder):
        if filename.endswith(".athdf"):
            filepath = os.path.join(dataFolder, filename)
            files.append(filepath)

    finalFile=files[-1]
    data=athena_read.athdf(finalFile)

    radius =np.asarray(constructDataArray(data['RootGridX1'][0], data['RootGridX1'][1], data['RootGridSize'][0],data['RootGridX1'][2]),dtype=np.float64)
    theta = np.asarray(constructDataArray(data['RootGridX2'][0], data['RootGridX2'][1], data['RootGridSize'][1],data['RootGridX2'][2]),dtype=np.float64)
    phi = np.asarray(constructDataArray(data['RootGridX3'][0], data['RootGridX3'][1], data['RootGridSize'][2],data['RootGridX3'][2]),dtype=np.float64)
    print(len(radius))

    density=data['rho']
    pressure=data['press']
    radialVel=data['vel1']
    newRadius = np.linspace(radius[0], radius[-1], resolution)
    print("1")
    interpolateDensity=np.asarray(interpolateVals(radius,theta,resolution, density[0]), dtype=np.float64)
    print("2")
    interpolatePressure = np.asarray(interpolateVals(radius,theta,resolution, pressure[0]), dtype=np.float64)
    print("3")
    interpolateRadialVel = np.asarray(interpolateVals(radius,theta, resolution,radialVel[0]), dtype=np.float64)
    print("4")

    if not verbose:
        np.asarray(density, dtype=np.float64).tofile(densityPath)
        np.asarray(pressure, dtype=np.float64).tofile(pressurePath)
        np.asarray(radialVel, dtype=np.float64).tofile(radialVelPath)
        np.asarray(radius, dtype=np.float64).tofile(radiusPath)
        np.asarray(phi, dtype=np.float64).tofile(phiPath)
        np.asarray(theta, dtype=np.float64).tofile(thetaPath)

        with open(paramsPath, 'w') as file:
            file.write("Initial Radius:"+str(radius[0])+"\n")
            file.write("Final Radius:"+str(radius[-1])+"\n")
            file.write("Resolution:"+str(resolution)+"\n")
    else:
        plt.polar()
        plt.title(f"Density over space")
        #plt.imshow(np.asarray(interpolateDensity), aspect='auto', cmap='inferno', origin='lower',extent=(theta[0], theta[-1], newRadius[0], newRadius[-1]),norm=LogNorm(vmin=1e-10, vmax=1e-6))
        im = plt.pcolormesh(theta, newRadius,np.transpose(np.asarray(interpolateDensity[:-1,:-1])), cmap='inferno', norm=LogNorm(vmin=1e-12,vmax=1e-4)) # , vmin=0,vmax=1)
        #im = plt.pcolormesh(theta, radius, np.transpose(np.asarray(density)), cmap='inferno',norm=LogNorm(vmin=1e-13, vmax=1e-7))  # , vmin=0,vmax=1)

        plt.colorbar(label="g/cm^3")
        plt.show()

def getHDF5data(fileName, dictName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(fileName) + "/" + str(fileName) + ".h5"
    with h5py.File(newPath, 'r') as file:
        return file[dictName][:]

def processExtraDataOctree(folderName, hdf5Name):
    pressure = getHDF5data(hdf5Name, "pressure")
    density = getHDF5data(hdf5Name, "density")
    radii = getHDF5data(hdf5Name, "radius")
    thetas = getHDF5data(hdf5Name, "theta")
    velocity = getHDF5data(hdf5Name, "velocity")
    minDensity = 1000000
    minPressure = 10 ** 11
    for i in range(len(thetas)):
        minDensity = min(minDensity, np.min(density[i]))
        minPressure = min(minPressure, np.min(pressure[i]))
    newMaxRadius = radii[-1] + .002 * (10 ** 13)
    edgeDensity=[]
    edgeRadii=[]
    edgePressure=[]
    edgeVr=[]
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(folderName)
    if not os.path.isdir(newPath):
        os.makedirs(newPath)
    densityFilePath = newPath + "/edgeRho.txt"
    pressureFilePath = newPath + "/edgePres.txt"
    velocityFilePath = newPath + "/edgeVr.txt"
    radiusFilePath = newPath + "/edgeR.txt"
    thetaFilePath = newPath + "/edgeTheta.txt"
    with open(densityFilePath, 'x') as file:
        for j in range(len(radii)):
            if density[-1][j]==minDensity:
                file.write(str(density[-1][j]/100) + "\n")
            else:
                file.write(str(density[-1][j]) + "\n")
        file.write(str(minDensity/100)+"\n")

    with open(pressureFilePath, 'x') as file:
        for j in range(len(radii)):
            if density[-1][j] == minDensity:
                file.write(str(pressure[-1][j] / 100) + "\n")
            else:
                file.write(str(pressure[-1][j]) + "\n")
        file.write(str(minPressure/100)+"\n")

    with open(velocityFilePath, 'x') as file:
        for j in range(len(radii)):
            file.write(str(velocity[-1][j]) + "\n")
        file.write(str(0) + "\n")

    with open(radiusFilePath, 'x') as file:
        for j in range(len(radii)):
            file.write(str(radii[j]) + "\n")
        file.write(str(newMaxRadius) + "\n")
    print("FIN")

def deleteIntensityCorrectly(intensityFile):

    with open(intensityFile, 'r') as file:
        lines=file.readlines()

    linesToKeep=[]
    for i in range(len(lines)):
        if (i//1200)%2==0:
            linesToKeep.append(lines[i])

    with open(intensityFile, 'w') as file:
        file.writelines(linesToKeep)


def main():
    #deleteIntensityCorrectly('locationCopy.txt')
    #processFinalFiles("data/storeDir2Restart","test",3000,verbose=True)
    #processFinalFilesAthdf("data/eightPercentFinalRestart", "eightPercentFinalRestart",3000,verbose=False)
    processFinalFilesAthdf2("data/sixteenPercentRestart1", "sixteen1", 2000, verbose=False)
    #plotRadiationEnergy("data/testNewRestart","testNew",1000)
    #storeArrays, maxVel=createDataGrid("lowerFloor100","a",2000,600,verbose=False)
    #print(len(maxVel))
    #makeBinaryFileVel(storeArrays, maxVel,'testBinaryVelocities')
    #lowerFloor("fullDataSixteenPercent", "lowerFloor100SixteenPercent", 100.0)

    '''
    directory = "data/3_13_24_finalRunStore100"

    # Define the naming scheme of the files
    prefix = "supernovaCollision.out1"
    extension = ".athdf"

    # Create an empty list to store the data from all files
    files = []

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            files.append(filepath)
    data1 = athena_read.athdf(files[0])
    print(data1)
    processFiles(files, 'supernovaCollisionFinalRun')
    '''
    return


if __name__=="__main__":
    main()







