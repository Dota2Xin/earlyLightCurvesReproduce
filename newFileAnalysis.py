import sys
import h5py
import numpy as np
import ast
import athena_read
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter
import warnings
import os
from scipy import optimize
import math
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from processFiles import getDataDict
from numba import njit
from scipy import interpolate
import h5py

#Global constants to use in program
a= 7.56*(10**-15)
mu=4.0/3
kB=1.3807*(10**-16)
mProton=1.6726*(10**-24)
#Root Finder from wolfram alpha solving x^3+4Bx-A^2=0
@njit
def findRoot(A,B):
    A=np.array(A,dtype=np.float64)
    B=np.array(B,dtype=np.float64)
    z3=((81*(A**4)+768*(B**3))**.5)+9*A*A
    numerator=(2**(1.0/3.0))*(z3**(2.0/3.0))-8*(3**(1.0/3.0))*B
    denominator=(6**(2.0/3.0))*(z3**(1.0/3.0))
    return numerator/denominator

@njit
def pressureEq(temp, rho):
    return (rho*kB*temp)/(mu*mProton)+(1/3)*a*(temp**4)

@njit
def getGasPress(temperature,density):
    return kB*temperature*density/(mu*mProton)

@njit
def getRadPress(temperature):
    return (a/3)*(temperature**4)
#Gets temperature by solving quartic P_total=a/3T^4+(rho*kB)/(mu*mProton)*T
@njit
def getTemperature(pressure, density):
    A=3*kB*density/(a*mu*mProton)#float(3*kB*density/(a*mu*mProton))
    B=3*pressure/a#float(3*pressure/a)
    y=findRoot(A,B)
    #warnings.filterwarnings("error")
    temp=(y**.5)*(((2*A/(y**(3.0/2.0))-1)**.5)-1)/2
    return temp

@njit
def getEntropy(temperature, density):
    return kB/(mu*mProton)*np.log((temperature**(3.0/2.0))/density)+(4.0/3)*a*(temperature**3)/density

@njit
def entropyRootEq(density,A,B,T):
    return np.log((T**1.5)/density)+A*(T**3)/density-B

@njit
def findRootEntropy(density, A,B, upperBound):
    upper=2*upperBound
    lower=0
    currentGuess=(upper-lower)/2
    currentValue=entropyRootEq(density,A,B,currentGuess)
    count=0
    while abs(currentValue/B) >.0001:
        count=count+1
        if currentValue>0:
            upper=currentGuess
        else:
            lower=currentGuess
        currentGuess=(upper-lower)/2+lower
        currentValue=entropyRootEq(density, A,B,currentGuess)
        if count>100 and count<110:
            print("STUCK")
            print(density)
            print(A)
            print(B)
            print(upperBound)
            print(upper)
            print(lower)
            print(currentGuess)
            print(currentValue)
    return currentGuess

@njit
def evolveTempStartTemp(entropy,initialTemperature, density):
    C=entropy
    A = 4.0 * a * mu * mProton / (3.0 * kB)
    B = C * mu * mProton / kB
    temperature = findRootEntropy(density, A, B, initialTemperature*1.1)
    return temperature

@njit
def evolveTempNuclearDecay(initialP,initialRho,epsilon0,tau,startTime,finalTime,steps, velocity,maxRadius, timeStuff):
    dt=finalTime/steps
    currentT=getTemperature(initialP,initialRho)
    currentS=getEntropy(currentT,initialRho)
    time=startTime

    for i in range(steps-1):
        time=time+dt
        #newRho=initialRho*((startTime/time)**3)
        radiusStore=maxRadius+(time-timeStuff)*velocity
        newRho=initialRho*((maxRadius/radiusStore)**3)
        nextT=evolveTempStartTemp(currentS,currentT,newRho)
        avT=(nextT+currentT)/2
        dS=(epsilon0)*(dt/avT)*np.exp(-1*(time-dt)/tau)
        currentS=currentS+dS
        currentT=nextT

    return currentT

@njit
def evolveEdgeAnalysis(density, velocity, pressure, minRadius,maxRadius, finalTime, firsts, epsilon0, tau):
    newDensity=np.copy(density[0:])
    newRadius=np.copy(density[0:])
    newTemperature=np.copy(density[0:])
    newVelocity=np.copy(velocity[0:])
    t0=minRadius/ (2.53 * (10 ** 9))
    for i in range(len(density)):
        #print(density[i])
        #if i==350:
            #break
        for j in range(len(density[0])):
            if firsts[j]>=20 and (density[i][j]>=.000001 or i>=300):
                newRadius[i][j]=maxRadius+(finalTime-i-t0)*velocity[i][j]
                newDensity[i][j]=density[i][j]*((maxRadius/(newRadius[i][j]))**3)
                newTemperature[i][j]=evolveTempNuclearDecay(pressure[i][j],density[i][j],epsilon0,tau,i+t0,finalTime,10000, velocity[i][j], maxRadius,i+t0)
            elif not (density[i][j]>=.000001 or i>=300):
                pass
            else:
                firsts[j]=firsts[j]+1
        print(i)
    return newRadius, newDensity, newTemperature, newVelocity

def processThetas(radius, density, temperature,velocity, oldDensity):
    newRadius=[[] for _ in range(len(density[0]))]
    newDensity=[[] for _ in range(len(density[0]))]
    newTemperature=[[] for _ in range(len(density[0]))]
    newVelocity= [[] for _ in range(len(density[0]))]
    for i in range(len(radius)):
        for j in range(len(radius[i])):
            if not (density[i][j] ==oldDensity[i][j]):
                newRadius[j].append(radius[i][j])
                newDensity[j].append(density[i][j])
                newTemperature[j].append(temperature[i][j])
                newVelocity[j].append(velocity[i][j])
    return newRadius, newDensity, newTemperature, newVelocity


def evolveEdgeFull(dataDict, minRadius,maxRadius, finalTime, epsilon0, tau, verbose=True):
    oldDensity=np.asarray(dataDict["density"])
    oldVelocity=np.asarray(dataDict["velocity"])
    oldPressure=np.asarray(dataDict["pressure"])
    firsts=[0]*len(oldDensity[0])
    radius, density, temperature, velocity=evolveEdgeAnalysis(oldDensity, oldVelocity, oldPressure, minRadius,maxRadius, finalTime,np.asarray(firsts), epsilon0, tau)

    print("DONE")
    radius, density, temperature, velocity=processThetas(radius, density, temperature,velocity, oldDensity)

    interpDensity, interpVelocity,interpTemperature, gridRadii, thetas=interpolateData(radius, density, velocity, 1000,temperature=temperature, verbose=True)
    #print(interpTemperature)
    interpPressure=np.copy(interpTemperature)
    pressure=[]
    for i in range(len(temperature)):
        pressure.append([])
        for j in range(len(temperature[i])):
            pressure[i].append(pressureEq(temperature[i][j],density[i][j]))
    for i in range(len(interpTemperature)):
        for j in range(len(interpTemperature[i])):
            interpPressure[i][j]=pressureEq(interpTemperature[i][j], interpDensity[i][j])
    if verbose:
        thetaIndex=[20,120,180]
        colorStore = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        plt.figure()
        plt.loglog()
        for i in range(len(thetaIndex)):
            currentIndex=thetaIndex[i]
            theta=thetas[currentIndex]
            plt.subplot(211)
            plt.loglog()
            plt.plot(gridRadii,interpDensity[currentIndex], label=f"Interp Theta: {theta}", color=colorStore[i])
            plt.plot(radius[currentIndex], density[currentIndex], label=f"Actual Theta: {theta}", color=colorStore[i],linestyle='--')
            plt.xlabel("Radius (cm)")
            plt.ylabel("Density (g/cm^3)")
            plt.title(f"Density vs Radius for Various Angles, time={finalTime}")
            plt.legend()
            plt.subplot(212)
            plt.loglog()
            plt.xlabel("Radius (cm)")
            plt.ylabel("Pressure (Ba)")
            plt.title(f"Pressure vs Radius for Various Angles, time={finalTime}")
            plt.plot(gridRadii,interpPressure[currentIndex], label=f"Theta: {theta}", color=colorStore[i])
            plt.plot(radius[currentIndex],pressure[currentIndex], label=f"Actual Theta: {theta}", color=colorStore[i],linestyle='--')
            plt.legend()

        plt.tight_layout()
        plt.show()

    return interpDensity, interpPressure,interpVelocity,gridRadii, thetas

def kineticEnergy(density, velocity):
    return .5*density*(velocity**2)

def momentum(density, velocity):
    return density*velocity

def mass(density, velocity):
    return density

def integrationFunc(chunkFunction, density,radius,thetas, velocity):
    integrationValue=0
    for i in range(len(density)-1):
        for j in range(len(density[i])-1):
            integrationValue+= 2*np.pi*chunkFunction(density[i][j], velocity[i][j])*(radius[i][j]**2)*np.sin(thetas[i])*(abs(radius[i][j+1]-radius[i][j]))*(thetas[i+1]-thetas[i])
    return integrationValue

@njit
def dynamicsAnalysis(density, velocity, pressure, minRadius,maxRadius, finalTime, firsts):
    newDensity=np.copy(density[0:])
    newRadius=np.copy(density[0:])
    newVelocity=np.copy(velocity[0:])
    newTemperature=np.copy(density[0:])
    t0=minRadius/ (2.53 * (10 ** 9))
    for i in range(len(density)):
        #print(density[i])
        for j in range(len(density[0])):
            if firsts[j]>=20 and (density[i][j]>=.000001 or i>=300):
                newRadius[i][j]=maxRadius+(finalTime-i-t0)*velocity[i][j]
                newDensity[i][j]=density[i][j]*((maxRadius/newRadius[i][j])**3)
                newVelocity[i][j]=velocity[i][j]
            elif not (density[i][j]>=.000001 or i>=300):
                pass
            else:
                firsts[j]=firsts[j]+1
        print(i)
    return newRadius, newDensity, newVelocity

def processThetasDynamic(radius, density,velocity, oldDensity):
    newRadius=[[] for _ in range(len(density[0]))]
    newDensity=[[] for _ in range(len(density[0]))]
    newVelocity=[[] for _ in range(len(density[0]))]
    for i in range(len(radius)):
        for j in range(len(radius[i])):
            if not (density[i][j] ==oldDensity[i][j]):
                newRadius[j].append(radius[i][j])
                newDensity[j].append(density[i][j])
                newVelocity[j].append(velocity[i][j])
    return newRadius, newDensity, newVelocity

def getDynamicData(dataDict, minRadius,maxRadius, finalTime, verbose=True, interpolate=False):
    oldDensity = np.asarray(dataDict["density"])
    oldVelocity = np.asarray(dataDict["velocity"])
    oldPressure = np.asarray(dataDict["pressure"])
    firsts = [0] * len(oldDensity[0])
    radius, density, velocity= dynamicsAnalysis(oldDensity, oldVelocity, oldPressure, minRadius, maxRadius,
                                                      finalTime, np.asarray(firsts))
    radius, density, velocity= processThetasDynamic(radius, density, velocity, oldDensity)
    thetas=np.linspace(0,1.4,len(density))
    print("Energy:"+str(integrationFunc(kineticEnergy, density, radius, thetas, velocity)/(.97*(10**51.0))))
    print("Momentum:" + str(integrationFunc(momentum, density, radius, thetas, velocity)/(1.789623*(10**33.0)*8.5*(10**8.0))))
    print("Mass:" + str(integrationFunc(mass, density, radius, thetas, velocity)/(1.789623*(10**33.0))))

    if verbose:
        interpolateData(radius, density, velocity, 1000)
        thetaIndex=[20,100,180]
        for index in thetaIndex:
            thetaValue=thetas[index]
            plt.plot(radius[index],density[index], label=f"Theta={thetaValue:.3f}")
        plt.loglog()
        plt.xlabel("Radius (cm)")
        plt.ylabel("Density g/cm^3")
        plt.title(f"Radius vs Density at {finalTime}")
        plt.legend()
        plt.show()

    if interpolate:
        return interpolateData(radius, density, velocity, 1000, verbose=False)
    return radius, density, velocity




def getSNRData(dataDict, lastFile, innerRadiusValue, maxRadius, minRadius, verbose=True):
    density=np.asarray(dataDict['density'])
    velocity=np.asarray(dataDict['velocity'])
    t0 = minRadius / (2.53 * (10 ** 9))
    finalTime=10000
    finalDensities=[[] for _ in range(len(density[0]))]
    finalVelocities=[[] for _ in range(len(density[0]))]
    finalRadii=[[] for _ in range(len(density[0]))]

    innerRadiusIndex=0
    data=athena_read.athdf(lastFile)
    densityLast=data['rho']
    velocityLast=data['vel1']
    radiusLast=data['x1f']
    for i in range(len(radiusLast)):
        if innerRadiusValue< radiusLast[i]:
            innerRadiusIndex=i
            break
    innerRadiusIndex=len(radiusLast)-innerRadiusIndex
    angleLast=data['x2f']
    firsts=[0]*len(density[0])

    for i in range(len(density)):
        for j in range(len(density[0])):
            if firsts[j]>=20 and (density[i][j]>=.000001 or i>=300):
                radiusValue = maxRadius + (finalTime- i-t0) * velocity[i][j]
                finalVelocities[j].append(velocity[i][j])
                finalRadii[j].append(radiusValue)
                finalDensities[j].append(density[i][j] * ((maxRadius/radiusValue)**3))
            elif not (density[i][j] >= .000001 or i >= 300):
                pass
            else:
                firsts[j] = firsts[j] + 1

    for i in range(innerRadiusIndex):
        for j in range(len(angleLast)-2):
            radiusValue=radiusLast[-i-1]+(finalTime-1000-t0)*velocityLast[0][j][-i-1]
            finalRadii[j].append(radiusValue)
            finalVelocities[j].append(velocityLast[0][j][-i-1])
            finalDensities[j].append(densityLast[0][j][-i-1]*((radiusLast[-i-1]/radiusValue)**3))

    indices=[20,100,180]

    for index in indices:
        thetaVal=angleLast[index]
        plt.plot(finalRadii[index], finalDensities[index], label=f"Theta={thetaVal}")
        plt.xlabel("Radius (cm)")
        plt.ylabel("Density (g/cm^3)")
        plt.title("Density vs Radius")
    plt.loglog()
    plt.legend()
    plt.show()
    return finalRadii, finalDensities, finalVelocities

def processSNRData(radius, density, velocity, folderName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(folderName)
    notOpened = True
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    else:
        notOpened = False
    densityFilePath = newPath + "/density.txt"
    velocityFilePath = newPath + "/radialVelocity.txt"
    radiusFilePath = newPath + "/radius.txt"
    filePaths=[densityFilePath,velocityFilePath, radiusFilePath]
    data=[density, velocity, radius]

    if notOpened:
        for i in range(len(filePaths)):
            with open(filePaths[i], 'w') as file:
                for j in data[i]:
                    file.write(str(j)+"\n")
    else:
        for i in range(len(filePaths)):
            with open(filePaths[i], 'a') as file:
                for j in data[i]:
                    file.write(str(j)+"\n")



def getFiles():
    # Set the directory path where the files are located
    directory = "data/3_13_24_finalSupernovaRunEvery10"

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
    return files


@njit
def cutData(maxRadius,minRadius, gridRadii):
    storeIndex=len(gridRadii)-1
    startIndex=0
    for i in range(len(gridRadii)):
        if gridRadii[i]<minRadius:
            #print("Hello")
            startIndex=i+1
        if gridRadii[i]>maxRadius:
            storeIndex=i
            break
    return gridRadii[startIndex:storeIndex], storeIndex

def interpSingle(gridRadii, realRadius, interpData, floorVal):
    currentMax=np.max(realRadius)
    currentMin=np.min(realRadius)
    #print(currentMax)
    #print(currentMin)
    cutRadii, cutIndex=cutData(currentMax,currentMin,gridRadii)
    interpFunction=interpolate.interp1d(realRadius, interpData, kind='linear', fill_value='interpolation')

    #print(cutRadii)
    #print(cutIndex)
    interpValues=np.asarray(interpFunction(cutRadii))

    fillValues=np.asarray([floorVal]*(len(gridRadii)-cutIndex))
    totalValues=np.concatenate((interpValues,fillValues))
    return totalValues

def plotMinMaxRadius(radius):
    minArray = []
    maxArray = []
    angels = np.linspace(0, 1.393, len(radius))
    for i in range(len(radius)):
        minArray.append(np.min(radius[i]))
        maxArray.append(np.max(radius[i]))
    plt.figure()
    plt.subplot(211)
    plt.plot(angels, minArray)
    plt.xlabel("Theta")
    plt.ylabel("Minimum Radius")
    plt.title("Minimum Radius vs Angle")
    plt.subplot(212)
    plt.plot(angels, maxArray)
    plt.xlabel("Theta")
    plt.ylabel("Maximum Radius")
    plt.title("Maximum Radius vs Angle")
    plt.tight_layout()
    plt.show()

def interpolateData(radius, density, velocity, gridResolution,temperature=[],verbose=True):
    minRadius=np.min(radius[0])
    maxRadius=radius[0][-1]
    minDensity=density[0][0]
    minTemperature=temperature[0][0]
    for i in range(len(radius)):
        maxRadius=max(np.max(radius[i]), maxRadius)
        minRadius=max(np.min(radius[i]),minRadius)
        minDensity=min(minDensity,np.min(density[i]))
        minTemperature=min(minTemperature,np.min(temperature[i]))
    #plotMinMaxRadius(radius)
    gridRadii=np.linspace(minRadius, maxRadius, gridResolution)
    interpDensity=[]
    interpVelocity=[]
    interpTemperature=[]
    densityFloor=minDensity*(10**-9)
    velocityFloor=0
    temperatureFloor=minTemperature*(10**-9)
    print(minRadius)
    print(maxRadius)
    for i in range(len(density)):
        interpDensity.append(interpSingle(gridRadii,radius[i],density[i], densityFloor))
        interpVelocity.append(interpSingle(gridRadii,radius[i],velocity[i],velocityFloor))
        if temperature !=[]:
            interpTemperature.append(interpSingle(gridRadii,radius[i],temperature[i],temperatureFloor))
    if verbose:
        indices = [20, 100, 180]
        thetaVals=np.linspace(0,1.4,len(interpDensity))
        plt.figure()
        for index in indices:
            thetaVal=thetaVals[index]
            plt.subplot(211)
            plt.plot(gridRadii, interpDensity[index], label=f"Interpolated, Theta={thetaVal:.2f}")
            plt.plot(radius[index], density[index], label=f"Actual Values, Theta={thetaVal:.2f}",linestyle= '--')
            plt.legend()
            plt.xlabel("Radius")
            plt.ylabel("Density")
            plt.title("Density vs Radius")
            plt.loglog()
            plt.subplot(212)
            plt.plot(gridRadii, interpVelocity[index], label=f"Interpolated, Theta={thetaVal:.2f}")
            plt.plot(radius[index], velocity[index], label=f"Actual Values, Theta={thetaVal:.2f}", linestyle='--')
            plt.legend()
            plt.xlabel("Radius")
            plt.ylabel("Velocity")
            plt.title("Velocity vs Radius")
        plt.tight_layout()
        plt.show()
    return interpDensity, interpVelocity, interpTemperature, gridRadii, np.linspace(0,1.393,len(density))

def makeHDF5data(density, velocity, radius,pressure, theta, folderName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(folderName)
    notOpened = True
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    else:
        notOpened = False

    filePath=newPath+"/"+str(folderName)+".h5"
    with h5py.File(filePath, 'w') as file:
        file.create_dataset('density', data=density)
        file.create_dataset('velocity', data=velocity)
        file.create_dataset('radius', data=radius)
        file.create_dataset('theta',data=theta)
        file.create_dataset('pressure',data=pressure)

def readHDF5data(fileName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(fileName)+"/"+str(fileName)+".h5"
    with h5py.File(newPath, 'r') as file:
        print("Dataset Names:")
        for dataset_name in file.keys():
            print(dataset_name)
            print(len(file[dataset_name]))
            print(file[dataset_name][:])  # Print the entire datasennhl;'t

def getHDF5data(fileName, dictName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/processedEdgeData/" + str(fileName) + "/" + str(fileName) + ".h5"
    with h5py.File(newPath, 'r') as file:
        return file[dictName][:]
@njit
def evolveEntropyNuclearDecay(initialP,initialRho,epsilon0,tau,startTime,finalTime,steps, velocity,maxRadius, timeStuff):
    dt=finalTime/steps
    currentT=getTemperature(initialP,initialRho)
    currentS=getEntropy(currentT,initialRho)
    time=startTime
    deltaS=0
    for i in range(steps-1):
        time=time+dt
        #newRho=initialRho*((startTime/time)**3)
        radiusStore=maxRadius+(time-timeStuff)*velocity
        newRho=initialRho*((maxRadius/radiusStore)**3)
        '''
        print("Integration:")
        print(newRho)
        print(currentT)
        print(currentS)
        print(initialP)
        print(initialRho)
        print(f"Newrho= {newRho}, currentT={currentT}, currentS={currentS} \n initialP={initialP}, initialRho={initialRho}")
        '''
        nextT=evolveTempStartTemp(currentS,currentT,newRho)
        avT=(nextT+currentT)/2
        dS=(epsilon0)*(dt/avT)*np.exp(-1*(time-dt)/tau)
        deltaS +=dS
        currentS=currentS+dS
        currentT=nextT

    return deltaS

def processEntropyGeneration(dataDict, minRadius,maxRadius, epsilon0Fake, tau, verbose=True):
    density = np.asarray(dataDict["density"])
    velocity = np.asarray(dataDict["velocity"])
    pressure = np.asarray(dataDict["pressure"])
    times=[250,350, 500, 750]
    angles=[0, 70, 150]
    massFractions=np.logspace(-4,-1,20)
    finalEvolutions=[7200, 43000, 86400]
    entropyJump=[]

    t0 = minRadius / (2.53 * (10 ** 9))
    for j in range(len(finalEvolutions)):
        entropyJump.append([])
        finalTime = finalEvolutions[j]
        for k in range(len(times)):
            time = times[k]
            entropyJump[j].append([])
            for l in range(len(angles)):
                angle=angles[l]
                entropyJump[j][k].append([])
                for i in range(len(massFractions)):
                    massFraction = massFractions[i]
                    deltaS = evolveEntropyNuclearDecay(pressure[time][angle], density[time][angle], epsilon0Fake*massFraction, tau, time + t0,
                                                                  finalTime, 10000, velocity[time][angle], maxRadius, time + t0)
                    entropyVals=deltaS
                    print("MADE IT")
                    entropyJump[j][k][l].append(entropyVals)

    thetas=np.linspace(0, 1.394, 199)
    #first plotting sequence
    timeIndex=0
    finalTimeIndex=1
    finalTimeVal=finalEvolutions[finalTimeIndex]
    timeVal=times[timeIndex]
    for i in range(len(angles)):
        angleVal=thetas[angles[i]]
        plt.plot(massFractions, entropyJump[finalTimeIndex][timeIndex][i], label=f'Theta: {angleVal:.3f}')
    plt.loglog()
    plt.legend()
    plt.xlabel("Mass Fraction")
    plt.ylabel("Total Entropy Generated After Evolution")
    plt.title(f"Entropy Generated vs Mass Fraction (Evolution Time= {finalTimeVal:.1f}s, Material That reached edge at {timeVal}s")
    plt.show()


def plotFinalSnapshot(data, radius, thetas, dataName, min, max, units):
    plt.polar()
    plt.title(f"{dataName} over space")
    im = plt.pcolormesh(thetas, radius, np.transpose(data[:-1,:-1]), cmap='inferno',
                        norm=LogNorm(vmin=min, vmax=max))  # , vmin=0,vmax=1)
    im.set_clim(min, max)
    plt.colorbar(label=units)
    plt.show()

def plotFinalSnapshotDouble(density, temperature, radius, thetas, dataName, min, max, units):
    plt.polar()
    for label in plt.gca().get_yticklabels():
        label.set_visible(False)

    for label in plt.gca().get_xticklabels():
        label.set_visible(False)
    scaleFactor=10*np.max(temperature)/np.max(density)
    #plt.title(f"Density and Temperature/{scaleFactor:.0f} over space")
    fullArray=np.concatenate((np.flip(np.transpose(temperature/scaleFactor),axis=1),np.transpose(density)),axis=1)
    fullThetaArray=np.concatenate((list(reversed(-1.0*thetas)),thetas))
    im = plt.pcolormesh(fullThetaArray, radius, fullArray, cmap='inferno',
                        norm=LogNorm(vmin=min, vmax=max))  # , vmin=0,vmax=1)
    im.set_clim(min, max)
    plt.colorbar(label=units)
    rTicks = plt.yticks()[0]
    for i in range(len(rTicks) - 1):
        r = rTicks[i]
        nums = i + 1
        plt.text(np.pi / 2, r + .05 * (10 ** 13.0), f'{nums}')
    plt.text(1.4, 2.7 * (10 ** 13.0), "$\\times \\textrm{10}^{\\textrm{13}}$ cm")
    plt.text(.9, 2.75 * (10 ** 13.0), f"t={8200}s", fontsize=8)
    print("MIN TEMP:")
    print(temperature[-10][60])
    #print(scaleFactor)
    plt.show()

def plotRadiusGraphDensity(thetas, thetaIndex, radii, density):
    for i in range(3):
        thetaVal=thetas[thetaIndex[i]]
        plt.plot(radii, density[thetaIndex[i]], label=f"Theta={thetaVal}")
        plt.legend()
        plt.xlabel("Radius (cm)")
        plt.ylabel("Density (g/cm^3)")
        plt.title("Density vs Radius for Three Angles")
    plt.loglog()
    plt.show()

def plotRadiusGraphTemperature(thetas, thetaIndex, radii, density,pressure):
    for i in range(3):
        thetaVal=thetas[thetaIndex[i]]
        densityTemp=density[thetaIndex[i]]
        pressureTemp=pressure[thetaIndex[i]]
        temps=[]
        for j in range(len(densityTemp)):
            temps.append(getTemperature(pressureTemp[j],densityTemp[j]))
        plt.plot(radii, temps, label=f"Theta={thetaVal}")
        plt.legend()
        plt.xlabel("Radius (cm)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Radius for Three Angles")
    plt.loglog()
    plt.show()

def getRadRatio(density,temperature):
    tempData=np.asarray(temperature)
    return 3.0*density*tempData*kB/(mu*mProton*a*(tempData**4.0))
def plotRadiusGraphRadRatio(thetas, thetaIndex, radii, density,pressure):
    for i in range(3):
        thetaVal=thetas[thetaIndex[i]]
        densityTemp=density[thetaIndex[i]]
        pressureTemp=pressure[thetaIndex[i]]
        temps=[]
        for j in range(len(densityTemp)):
            temps.append(getTemperature(pressureTemp[j],densityTemp[j]))

        plt.plot(radii, getRadRatio(densityTemp,temps), label=f"Theta={thetaVal}")
        plt.legend()
        plt.xlabel("Radius (cm)")
        plt.ylabel("P_gas/P_rad")
        plt.title("P_gas/P_rad vs Radius for Three Angles")
    plt.loglog()
    plt.show()

def textify2D(array, folderName, fileName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(folderName)
    if not os.path.isdir(newPath):
        os.makedirs(newPath)
    filePath= newPath + "/"+fileName+".txt"
    with open(filePath, 'x') as file:
        for i in range(len(array)):
            for j in range(len(array[0])):
                file.write(array[i][j]+"\n")

def textify1D(array, folderName, fileName):
    newPath = "C:/Users/gabri/PycharmProjects/larsDataAnalysis/octreeInputData/" + str(folderName)
    if not os.path.isdir(newPath):
        os.makedirs(newPath)
    filePath= newPath + "/"+fileName+".txt"
    with open(filePath, 'x') as file:
        for i in range(len(array)):
            file.write(str(array[i])+"\n")

def processHDF5(startName, endName):
    pressure = getHDF5data(startName, "pressure")
    # pressure = getHDF5data("fullDataOnePercent2_18_25", "pressure")
    density = getHDF5data(startName, "density")
    radii = getHDF5data(startName, "radius")
    thetas = getHDF5data(startName, "theta")
    velocity = getHDF5data(startName, "velocity")
    pres=[]
    dens=[]
    vr=[]
    radius=[]
    theta=[]
    for i in range(len(density)):
        for j in range(len(density[0])):
            pres.append(pressure[i][j])
            vr.append(velocity[i][j])
            dens.append(density[i][j])
            radius.append(radii[j])
            theta.append(thetas[i])
    textify1D(pres,endName,"pres")
    textify1D(dens,endName,'rho')
    textify1D(vr,endName,'vr')
    textify1D(radius,endName,'r')
    textify1D(theta,endName,'theta')
    #textify1D(theta,'finalInputDataOnePercent','theta')

def plotRadRatio(radii, radiusValue, density, pressure,thetas):
    radialIndex = 0
    for i in range(len(radii)):
        if radii[i] > radiusValue:
            radialIndex = i
            break

    pRatio = []
    for i in range(len(thetas)):
        temperature=getTemperature(pressure[i][radialIndex],density[i][radialIndex])
        pRatio.append(a*(temperature**4.0)/(3.0*pressure[i][radialIndex]))
    timeVal=8200
    plt.plot(thetas,pRatio, label=f"t={timeVal}s")
    plt.xlabel("Thetas (rad)")
    plt.title("Radiation Pressure/Total Pressure")
    plt.legend()
    plt.text(0.0,.2, "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=.02$",fontsize=10)
    plt.text(0.0, (.1), "$\\textrm{R=1.5}\\times\\textrm{ 10}^{13}$ cm", fontsize=10)
    plt.show()

def plotEvolvedTemp(radii, density, temperature,velocity, thetas):
    tauRadiis=[]
    tempCopy=np.copy(temperature)
    for i in range(len(thetas)):
        tau=0
        for j in range(len(radii)-1):
            tau+=density[i][-j-1]*(radii[-j-1]-radii[-j-2])*.2
            tempCopy[i][-j-1]=0.0
            if velocity[i][-j-1]>0 and tau>(3*(10**10.0))/velocity[i][-j-1]:
                break
    #mynorm = matplotlib.colors.LogNorm(vmin=1e4, vmax=3e4)
    fig = plt.figure()
    plt.polar()
    rSun = 6.956 * (10 ** 10.0)
    im = plt.pcolormesh(thetas, radii / (rSun), np.transpose(tempCopy)[:-1, :-1],
                        cmap='inferno',
                        vmin=1e3, vmax=2e4, shading='flat')
    cbar = plt.colorbar(im)
    plt.tight_layout()
    plt.show()