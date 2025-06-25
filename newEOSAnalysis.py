import sys
import h5py
import numpy as np
import athena_read
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter
import warnings
from scipy.special import sph_harm
import os
from scipy import optimize
from numba import njit
from mathFunctions import *
import math
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.animation as animation


#Global constants to use in program
a= 7.56*(10**-15)
mu=4.0/3
kB=1.3807*(10**-16)
mProton=1.6726*(10**-24)
radDegree=360/(2*np.pi)
#Root Finder from wolfram alpha solving x^3+4Bx-A^2=0
def findRoot(A,B):
    A=np.array(A,dtype=np.float64)
    B=np.array(B,dtype=np.float64)
    z3=((81*(A**4)+768*(B**3))**.5)+9*A*A
    numerator=(2**(1.0/3.0))*(z3**(2.0/3.0))-8*(3**(1.0/3.0))*B
    denominator=(6**(2.0/3.0))*(z3**(1.0/3.0))
    return numerator/denominator

def pressureEq(temp, rho):
    return (rho*kB*temp)/(mu*mProton)+(1/3)*a*(temp**4)
def getGasPress(temperature,density):
    return kB*temperature*density/(mu*mProton)

def getRadPress(temperature):
    return (a/3)*(temperature**4)
#Gets temperature by solving quartic P_total=a/3T^4+(rho*kB)/(mu*mProton)*T
def getTemperature(pressure, density):
    A=3*kB*density/(a*mu*mProton)#float(3*kB*density/(a*mu*mProton))
    B=3*pressure/a#float(3*pressure/a)
    y=findRoot(A,B)
    warnings.filterwarnings("error")
    try:
        temp=(y**.5)*(((2*A/(y**(3.0/2.0))-1)**.5)-1)/2
        return temp
    except:
        print("Exception:")
        print(f"Pval={pressure}")
        print(f"density={density}")
        print(f"A={A}")
        print(f"B={B}")
        print(f"y={y}")

def getEntropy(temperature, density):
    return kB/(mu*mProton)*np.log((temperature**(3.0/2.0))/density)+(4.0/3)*a*(temperature**3)/density

#lets you plot all the major thermodynamic parameters for our system have to make specific functions for something else
def plotEdge(files, times, type, typeName, typeUnits):
    plt.figure()
    if type=="Temperature":
        for time in times:
            data = []
            dataFile = athena_read.athdf(files[time])
            finalIndex = len(dataFile['x1f']) - 2
            finalTheta = len(dataFile['x2f']) - 40
            for i in range(finalTheta):
                density=dataFile['rho'][0][i][finalIndex]
                pressure=dataFile['press'][0][i][finalIndex]
                data.append(getTemperature(pressure,density))
            times2 = [35, 50, 75, 99]
            time2=times2[time]
            plt.plot(radDegree*dataFile['x2f'][0:finalTheta],data[0:finalTheta], label=f"t={time2*10} s")
        #plt.title(f"{typeName} vs Theta for various times")
        plt.ylabel(f"{typeName}({typeUnits})")
        plt.text(.05, (2 * (10 ** 4.0)), "$\\textrm{r}_{\\rm{out}}=\\rm{10.8}$ R$_{\odot}$", fontsize=10)
        #plt.text(.05, (3.5 * (10 ** 4.0)), "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=0$", fontsize=10)
    elif type=="Entropy":
        for time in times:
            data = []
            dataFile = athena_read.athdf(files[time])
            finalIndex = len(dataFile['x1f']) - 2
            finalTheta = len(dataFile['x2f']) - 40
            for i in range(finalTheta):
                density = dataFile['rho'][0][i][finalIndex]
                pressure = dataFile['press'][0][i][finalIndex]
                data.append(getEntropy(getTemperature(pressure, density),density))
            plt.plot(radDegree*dataFile['x2f'][0:finalTheta],data[0:finalTheta], label=f"t={time*10} s")
        plt.title(f"{typeName} vs Theta for various times")
        plt.ylabel(f"{typeName}({typeUnits})")
    else:
        for time in times:
            data = []
            dataFile = athena_read.athdf(files[time])
            finalIndex = len(dataFile['x1f']) - 2
            finalTheta = len(dataFile['x2f']) - 40
            for i in range(finalTheta):
                data.append(dataFile[type][0][i][finalIndex])
            times2 = [35, 50, 75, 99]
            time2=times2[time]
            plt.plot(radDegree*dataFile['x2f'][0:finalTheta],data[0:finalTheta], label=f"t={time2*10} s")
        plt.text(45.8,(3*(10**-6.0)),"$\\textrm{r}_{\\rm{out}}=\\rm{10.8}$ R$_{\odot}$",fontsize=10)
        #plt.text(.8, (5.5 * (10 ** -6.0)), "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=0$",fontsize=10)
        #plt.title(f"{typeName} vs Theta for various times")
        plt.ylabel(f"{typeName}({typeUnits})")
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Theta (degrees)")
    plt.show()

def plotRamRatio(files, times):
    for time in times:
        data=[]
        dataFile=athena_read.athdf(files[time])
        finalIndex = len(dataFile['x1f']) - 2
        finalTheta = len(dataFile['x2f']) - 2
        for i in range(finalTheta):
            ramPressure=.5*dataFile['rho'][0][i][finalIndex]*(dataFile['vel1'][0][i][finalIndex]**2+dataFile['vel2'][0][i][finalIndex]**2)
            pressure=dataFile['press'][0][i][finalIndex]
            data.append(pressure/ramPressure)
        plt.plot(dataFile['x2f'][0:finalTheta],data, label=f"t={time*10} s")
    plt.text(0.0, (4 * (10 ** -6.0)), "$\\textrm{R=10.8}$ R$_{\odot}$", fontsize=10)
    #plt.text(0.0, (9 * (10 ** -6.0)), "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=0$",fontsize=10)
    #plt.title(f"Pressure/Ram Pressure vs Theta for various times")
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Theta (rad)")
    plt.ylabel(f"Pressure/Ram Pressure")
    plt.show()

def plotRadRatio(files, times):
    colorStore=["tab:blue","tab:orange", "tab:green", "tab:red"]
    times2=[35,50,75,99]
    for j in range(len(times)):
        time2=times2[j]
        dataGas=[]
        dataRad=[]
        dataFile=athena_read.athdf(files[times[j]])
        finalIndex = len(dataFile['x1f']) - 2
        finalTheta = len(dataFile['x2f']) - 40
        for i in range(finalTheta):
            density=dataFile['rho'][0][i][finalIndex]
            pressure=dataFile['press'][0][i][finalIndex]
            temperature=getTemperature(pressure,density)
            #gasRatio=getGasPress(temperature,density)/pressure
            radRatio=getRadPress(temperature)/pressure
            dataRad.append(radRatio)
            #dataGas.append(gasRatio)
        plt.plot(radDegree*dataFile['x2f'][0:finalTheta],dataRad[0:finalTheta], label=f"t={time2*10} s", color=colorStore[j])
        #plt.plot(dataFile['x2f'][0:finalTheta], dataGas, label=f"Gas Ratio, t={time * 10}s", color=colorStore[j], linestyle='--')
    #plt.title(f"Pressure Ratios vs Theta for various times")
    #plt.yscale('log')
    plt.text(0.0, .1, "$$\\textrm{r}_{\\rm{out}}=\\rm{10.8}$ R$_{\odot}$", fontsize=10)
    #plt.text(0.0,.2, "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=0$",fontsize=10)
    plt.ylabel(f"Radiation Pressure/Total Pressure")
    plt.xlabel("Theta (degrees)")
    plt.legend()
    plt.show()

def plotInnerRatio(files):
    dataGas = []
    dataRad = []
    times=[]
    for i in range(len(files)-1):
        dataFile=athena_read.athdf(files[i+1])
        density = dataFile['rho'][0][0][0]
        pressure = dataFile['press'][0][0][0]
        temperature = getTemperature(pressure, density)
        gasRatio = getGasPress(temperature, density) / pressure
        radRatio = getRadPress(temperature) / pressure
        dataGas.append(gasRatio)
        dataRad.append(radRatio)
        times.append(i*10)
    plt.plot(times,dataGas, color="tab:blue", label="Gas Ratio", linestyle="--")
    plt.plot(times,dataRad,color="tab:blue", label="Rad Ratio")
    plt.xlabel("Time")
    plt.ylabel("P_gas/rad /P_total")
    plt.legend()
    plt.title("Pressure Ratios at inner boundaries vs time")
    plt.show()




def plotGasEntropy(files, times):
    for time in times:
        data=[]
        dataFile=athena_read.athdf(files[time])
        finalIndex = len(dataFile['x1f']) - 2
        finalTheta = len(dataFile['x2f']) - 2
        for i in range(finalTheta):
            pressure=dataFile['press'][0][i][finalIndex]
            density=dataFile['rho'][0][i][finalIndex]
            data.append(pressure/(density**(5.0/3.0)))
        plt.plot(dataFile['x2f'][0:finalTheta],data, label=f"t={time*10}s")
    plt.title(f"P/rho^(5/3) vs Theta for various times")
    plt.yscale('log')
    plt.legend()
    plt.ylabel(f"P/rho^(5/3)")
    plt.show()

def plotVelocities(files, times):
    for j in times:
        rVelocityData = []
        thetaVelocityData=[]
        dataFile = athena_read.athdf(files[j])
        finalIndex = len(dataFile['x1f']) - 2
        finalTheta = len(dataFile['x2f']) - 2
        for i in range(finalTheta):
            rVelocityData.append(dataFile['vel1'][0][i][finalIndex])
            thetaVelocityData.append(dataFile['vel2'][0][i][finalIndex])
        plt.subplot(211)
        plt.plot(dataFile['x2f'][0:finalTheta], rVelocityData, label=f"Time={(j*10) }s")
        plt.legend()
        plt.xlabel('Theta (rad)')
        plt.ylabel("Radial Velocity (cm/s)")
        plt.title("Radial Velocity over Angle for various Times")
        plt.subplot(212)
        plt.plot(dataFile['x2f'][0:finalTheta], thetaVelocityData, label=f"Time={(j*10) }s")
        plt.legend()
        plt.xlabel('Theta (rad)')
        plt.ylabel("ThetaVelocity (cm/s)")
        plt.title("Theta Velocity over Angle for various Times")
    plt.tight_layout()
    plt.show()

def plotThetaVelocities(files, times):
    fig = plt.figure()
    for j in times:
        rVelocityData = []
        thetaVelocityData=[]
        dataFile = athena_read.athdf(files[j])
        finalIndex = len(dataFile['x1f']) - 2
        finalTheta = len(dataFile['x2f']) - 40
        for i in range(finalTheta):
            rVelocityData.append(dataFile['vel1'][0][i][finalIndex])
            thetaVelocityData.append(dataFile['vel2'][0][i][finalIndex]/100000.0)
        #plt.subplot(211)
        #plt.plot(dataFile['x2f'][0:finalTheta], rVelocityData, label=f"Time={(j*10) }s")
        #plt.legend()
        #plt.xlabel('Theta (rad)')
        #plt.ylabel("Radial Velocity (cm/s)")
        #plt.title("Radial Velocity over Angle for various Times")
        #plt.subplot(212)
        times2=[35,50,75,99]
        time2=times2[j]
        plt.plot(radDegree*np.asarray(dataFile['x2f'][0:finalTheta]), thetaVelocityData[0:finalTheta], label=f"t={(time2*10) } s")
        plt.legend()
        plt.xlabel('Theta (degrees)')
        plt.ylabel("Theta Velocity (km/s)")
        #plt.title("Theta Velocity over Angle for various Times")
    #plt.tight_layout()
    plt.text(45.8, -50, "$\\textrm{r}_{\\rm{out}}=\\rm{10.8}$ R$_{\odot}$", fontsize=10)
    #plt.text(.8, (-.4 * (10 **  7.0)), "$\\textrm{M}_{\\textrm{Ni}}/\\textrm{M}_{\\textrm{tot}}=0$",fontsize=10)
    print(fig.get_size_inches(), fig.dpi, fig.get_size_inches() * fig.dpi)
    plt.show()


def entropyRootEq(density,A,B,T):
    return np.log((T**1.5)/density)+A*(T**3)/density-B

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
        if count>100:
            print("####################################")
            print(f"Density:{density}")
            print(f"A:{A}")
            print(f"B:{B}")
            print(f"Upper Bound Passed: {upperBound}")
            print(f"Upper:{upper}")
            print(f"Lower:{lower}")
            print(f"CurrentVal:{currentValue}")
            print(f"Current Guess:{currentGuess}")
            print("####################################")
    return currentGuess

def evolveTemp(initialPressure, density, initialDensity):
    initialTemperature=getTemperature(initialPressure,initialDensity)
    C=getEntropy(initialTemperature,initialDensity)
    A = 4.0 * a * mu * mProton / (3.0 * kB)
    B = C * mu * mProton / kB
    temperature = findRootEntropy(density, A, B, initialTemperature)
    return temperature


def massIntegration(density,radius,thetas, velocity):
    massValue=0
    energyValue=0
    for i in range(len(density)-1):
        for j in range(len(density[i])-1):
            integrationChunk=density[i][j]*(radius[i][j]**2)*np.sin(thetas[i])*(abs(radius[i][j+1]-radius[i][j]))*(thetas[i+1]-thetas[i])
            massValue=massValue+integrationChunk
            energyChunk=(velocity[i][j])*density[i][j]*(radius[i][j]**2)*np.sin(thetas[i])*(abs(radius[i][j+1]-radius[i][j]))*(thetas[i+1]-thetas[i])
            energyValue=energyValue+energyChunk
    print(energyValue/(1.789623*(10.0**33)*(10.0**8.0)*8.5))
    return 2*np.pi*massValue/(1.789623*(10.0**33))
#calculate optical depth vs radius assume density is 0 past end of density given.
def tauIntegration(density,radius):
    tauValues=[0]
    totalSum=0
    for i in range(len(density)-2):
        integrationChunk=.2*(density[len(density)-1-i]+.5*(density[len(density)-2-i]-density[len(density)-1-i]))\
                         *abs(radius[len(radius)-2-i]-radius[len(radius)-1-i])
        totalSum=totalSum+integrationChunk
        #print(integrationChunk)
        #print(totalSum)
        tauValues.append(totalSum)
    return tauValues[::-1]
def evolveNoRad(files,finalTime):
    dataFile = athena_read.athdf(files[0])
    finalIndex = len(dataFile['x1f']) - 2
    finalTheta = len(dataFile['x2f']) - 2
    radius = [[] for _ in range(finalTheta)]
    density = [[] for _ in range(finalTheta)]
    temperature = [[] for _ in range(finalTheta)]
    velocity = [[] for _ in range(finalTheta)]
    firsts = [0] * finalTheta
    thetas = dataFile['x2f'][0:finalTheta]
    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    for i in range(len(files)):
        dataFile = athena_read.athdf(files[i])
        for j in range(finalTheta):
            densityInitial = dataFile['rho'][0][j][finalIndex]
            if densityInitial < .000001 and i < 30:
                pass
            else:
                if firsts[j] < 3:
                    firsts[j] = firsts[j] + 1
                else:
                    velocityCalc = dataFile['vel1'][0][j][finalIndex]
                    radiusCalc = dataFile['x1f'][finalIndex] + (finalTime - i * 10 - t0) * velocityCalc
                    densityCalc = densityInitial * (dataFile['x1f'][finalIndex] / radiusCalc) ** 3
                    temperatureCalc = evolveTemp(dataFile['press'][0][j][finalIndex], densityCalc, densityInitial)
                    radius[j].append(radiusCalc)
                    velocity[j].append(velocityCalc)
                    density[j].append(densityCalc)
                    temperature[j].append(temperatureCalc)
        print("SHIMMY EH SHIMMY AH "+str(i))
    return finalIndex, finalTheta,radius, density, temperature,velocity,t0, thetas
def evolveEdge(files, finalTime):
    ###############Calculates the final state of the material that leaves the simulation##############
    finalIndex, finalTheta, radius, density, temperature, velocity, t0, thetas=evolveNoRad(files,finalTime)

    print("Mass Value:" + str(massIntegration(density, radius, thetas, velocity)))
    ###########Plotting###########
    plt.figure()
    c = 3 * (10 ** 10)
    plt.loglog()
    indexPlotting=[60,90,180]
    colorStore = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j in range(len(indexPlotting)):
        i=indexPlotting[j]
        thetaValue=thetas[i]

        plt.subplot(311)
        plt.loglog()
        plt.plot(radius[i],density[i], label=f"Theta: {thetaValue:.3f}",color=colorStore[j])
        plt.xlabel("Radius (cm)")
        plt.ylabel("Density (g/cm^3)")
        plt.title(f"Density vs Radius at t={finalTime}")
        plt.legend()
        plt.subplot(312)
        plt.loglog()
        plt.plot(radius[i], temperature[i], label=f"Theta: {thetaValue:.3f}",color=colorStore[j])
        plt.xlabel("Radius (cm)")
        plt.ylabel("Temperature (K)")
        plt.title(f"Temperature vs Radius at t={finalTime}")
        plt.legend()
        plt.subplot(313)
        plt.yscale('log')
        plt.plot(radius[i], c / np.asarray(velocity[i]), label=f'c/v, Theta:{thetaValue:.2f}',color=colorStore[j], linestyle='--')
        plt.plot(radius[i], c / (4*np.asarray(velocity[i])), color=colorStore[j],linestyle='dotted')
        tauValues = tauIntegration(density[i], radius[i])
        plt.plot(list(reversed(radius[i][0:len(radius[i])-1])), tauValues, label=f'Tau, Theta:{thetaValue:.2f}',color=colorStore[j])
        plt.xlabel("Radius")
        plt.title(f"Optical Thickness and c/v vs Radius for various theta (t={finalTime}s)")
        plt.legend(fontsize='x-small')
    plt.tight_layout()
    plt.legend()
    plt.show()
def findShockIndices(a,k,density,pressure):
    shockIndex=0
    endIndex=0
    for i in range(len(density)-k):
        if density[i+k]/density[i]>a and shockIndex==0:
            shockIndex=i+k
            count=0
            while pressure[i+count]>5.1*(10**8):
                count=count+1
            endIndex=i+count-1
            break
    return shockIndex-k, shockIndex, endIndex
def findShock(dataFile):
    density=dataFile['rho'][0][0]
    radius=dataFile['x1f']
    pressure=dataFile['press'][0][0]
    startIndex, shockIndex, endIndex=findShockIndices(2,3,density,pressure)
    densityJump=density[shockIndex]/density[startIndex]
    pressureJump=pressure[shockIndex]/pressure[startIndex]
    standoffDistance=radius[endIndex]-radius[startIndex]

    return densityJump, pressureJump, standoffDistance

def analyzeShocksAllTimes(files):
    times=[]
    densityJumps=[]
    pressureJumps=[]
    standoffDistances=[]
    for i in range(len(files)-3):
        data=athena_read.athdf(files[i+3])
        times.append(i*10+20)
        densityJump, pressureJump, standoffDistance=findShock(data)
        densityJumps.append(densityJump)
        pressureJumps.append(pressureJump)
        standoffDistances.append(standoffDistance)
    plt.figure()
    plt.subplot(311)
    plt.plot(times, densityJumps)
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio of Post/Pre shock density")
    plt.title("Density Ratio vs Time")
    plt.subplot(312)
    plt.plot(times, pressureJumps)
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio of Post/Pre shock pressure")
    plt.title("Pressure Ratio vs Time")
    plt.subplot(313)
    plt.plot(times, standoffDistances)
    plt.xlabel("Time (s)")
    plt.yscale("log")
    plt.ylabel("Standoff Distance (cm)")
    plt.title("Standoff Distance vs Time")
    plt.tight_layout()
    plt.show()

def analyzePhiErrors(dataFile):
    densityRatio=[]
    data=athena_read.athdf(dataFile)
    thetas=data['x2f']
    phi=data['x3f']
    for i in range(len(phi)-1):
        densityRatio.append([])
        for j in range(len(thetas)-1):
            densityRatio[i].append(data['rho'][i][j][300]/data['rho'][0][0][300])
    plt.pcolormesh(thetas[0:len(thetas)-1],phi[0:len(phi)-1],densityRatio)
    plt.xlabel("Thetas")
    plt.ylabel("Phi")
    plt.colorbar()
    plt.title("Ratio of Pressure(theta, phi)/Pressure(0,0) at fixed radius and time")
    plt.show()

def compareFiles(file1, file2):
    dataFile1=athena_read.athdf(file1)
    dataFile2=athena_read.athdf(file2)
    finalTheta=len(dataFile1['x2f'])-2
    finalIndex=len(dataFile1['x1f'])-2
    thetas=dataFile1['x2f'][0:finalTheta]
    data1=[]
    data2=[]
    for i in range(finalTheta):
        density1=dataFile1['rho'][0][i][finalIndex]
        pressure1=dataFile1['press'][0][i][finalIndex]
        temperature1=getTemperature(pressure1,density1)
        density2 = dataFile2['rho'][0][i][finalIndex]
        pressure2 = dataFile2['press'][0][i][finalIndex]
        temperature2=(pressure2*3/a)**.25
        data1.append(temperature1)
        data2.append(temperature2)
    plt.plot(thetas,data1,label="New EOS")
    plt.plot(thetas,data2, label="Old EOS")
    plt.xlabel("Theta (rad)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature vs Angle at Edge Comparison, t=600")
    plt.legend()
    plt.show()

def compareEvolve(files):
    initialTimes=[30,60,90]
    colors=["tab:blue","tab:orange","tab:green"]
    finalTime=100000
    theta=120
    dataFile=athena_read.athdf(files[0])
    thetaVal=dataFile['x2f'][theta]
    count=0
    for i in initialTimes:
        dataFile=athena_read.athdf(files[i])
        finalTheta = len(dataFile['x2f']) - 2
        finalIndex = len(dataFile['x1f']) - 2
        times=np.linspace(i*10,finalTime,1000)
        tempDataAdiabatic=[]
        tempDataRad=[]
        tempDataGas=[]
        time=i*10
        for j in times:
            densityInitial=dataFile['rho'][0][theta][finalIndex]
            velocityCalc = dataFile['vel1'][0][theta][finalIndex]
            radiusCalc = dataFile['x1f'][finalIndex] + (j - i * 10) * velocityCalc
            densityCalc=densityInitial*(dataFile['x1f'][finalIndex]/radiusCalc)**3
            temperatureCalc = evolveTemp(dataFile['press'][0][theta][finalIndex], densityCalc, densityInitial)
            initialTemp=getTemperature(dataFile['press'][0][theta][finalIndex],densityInitial)
            tempCalcRad=initialTemp*((i*10)/j)
            tempDataAdiabatic.append(temperatureCalc)
            tempDataRad.append(tempCalcRad)
            tempDataGas.append(initialTemp*(((i*10)/j)**2))
        plt.plot(times,tempDataAdiabatic,label=f"Root Find Evolve, Time={time}", color=colors[count])
        plt.plot(times, tempDataRad,label=f"Radiation Dominated Evolve, Time={time}", color=colors[count], linestyle='--')
        plt.plot(times, tempDataGas, label=f"Gas Dominated Evolve, Time={time}", color=colors[count],linestyle='dotted')
        count=count+1
    plt.loglog()
    plt.ylabel("Temperature")
    plt.xlabel("Time")
    plt.title(f"Temperature vs Final Time for different evolution methods and Starting Times, Theta={thetaVal:.2f}")
    plt.legend()
    plt.show()

def plotInjected(files):
    densities=[]
    temperatures=[]
    times=[]
    for i in range(len(files)):
        dataFile=athena_read.athdf(files[i])
        density=dataFile['rho'][0][0][0]
        pressure=dataFile['press'][0][0][0]
        temperature=getTemperature(pressure,density)
        densities.append(density)
        temperatures.append(temperature)
        times.append(i*10)
    plt.figure()
    plt.subplot(211)
    plt.yscale('log')
    plt.plot(times,densities)
    plt.xlabel("Time(s)")
    plt.ylabel("Rho")
    plt.title("Density vs Time at injection")
    plt.subplot(212)
    plt.yscale('log')
    plt.plot(times, temperatures)
    plt.xlabel("Time(s)")
    plt.ylabel("Temp")
    plt.title("Temperature vs Time at injection")
    plt.tight_layout()
    plt.show()

def tDot(T,time,epsilon0,tau,A,B):
    nuclearTerm=epsilon0*np.exp(-1*time/tau)
    numerator=nuclearTerm-A*(T**4)*(time**2)-B*(T/time)
    denominator=A*(T**3)*(time**3)+(3/2)*B
    return numerator/denominator
def tDotNew(time, T, epsilon0, tau,t0,initialDensity):
    rho=initialDensity*((t0/time)**3)
    gasPres=getGasPress(T,rho)
    radPres=getRadPress(T)
    radioactivity=epsilon0*np.exp(time/tau)
    return T*(rho*radioactivity-(1/time)*(12*radPres+3*gasPres))/(1.5*gasPres+12*radPres)

# Takes in function and initial conditions (positions) as well as final time you go to and amount of steps you want and has a verbose command at
def tempRK4Integrator(f, x0, final, steps,initialTime,epsilon0,tau,startDensity, verbose=False):
    dt = (final-initialTime) / steps
    outputValues = []
    currentX = x0
    outputValues.append(currentX)
    time=initialTime
    A=(12*a/(startDensity*((initialTime)**3)))
    B=kB/(mu*mProton)
    for i in range(steps):
        k1 = dt * np.asarray(f(currentX,time,epsilon0,tau,A,B))
        k2 = dt * np.asarray(f(currentX + k1 / 2,time,epsilon0,tau,A,B))
        k3 = dt * np.asarray(f(currentX + k2 / 2,time,epsilon0,tau,A,B))
        k4 = dt * np.asarray(f(currentX + k3,time,epsilon0,tau,A,B))
        currentX = currentX + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        time=time+dt
        outputValues.append(currentX)
    if verbose:
        return outputValues
    else:
        return outputValues[-1]

def evolveTempStartTemp(entropy,initialTemperature, density):
    C=entropy
    A = 4.0 * a * mu * mProton / (3.0 * kB)
    B = C * mu * mProton / kB
    temperature = findRootEntropy(density, A, B, initialTemperature*1.1)
    return temperature

def evolveTempNuclearDecay(initialP,initialRho,epsilon0,tau,startTime,finalTime,steps):
    dt=finalTime/steps
    currentT=getTemperature(initialP,initialRho)
    currentS=getEntropy(currentT,initialRho)
    time=startTime
    outputValues=[currentT]
    for i in range(steps-1):
        '''
        time = time + dt
        newRho = initialRho * ((startTime / time) ** 3)
        dS=(epsilon0*np.exp(-1*(time-dt)/tau))*(dt/currentT)
        currentS=currentS+dS
        A = 4.0 * a * mu * mProton / (3.0 * kB)
        B = currentS * mu * mProton / kB
        currentT=findRootEntropy(newRho,A,B,currentT*1.1)
        outputValues.append(currentT)
        '''

        time=time+dt
        newRho=initialRho*((startTime/time)**3)
        nextT=evolveTempStartTemp(currentS,currentT,newRho)
        avT=(nextT+currentT)/2
        dS=(epsilon0)*(dt/avT)*np.exp(-1*(time-dt)/tau)
        currentS=currentS+dS
        currentT=nextT
        outputValues.append(currentT)

    return outputValues[-1]
def evolveWithRadiation(files, tau, epsilon0):
    ###############Calculates the final state of the material that leaves the simulation##############
    dataFile = athena_read.athdf(files[0])
    finalIndex = len(dataFile['x1f']) - 2
    finalTheta = len(dataFile['x2f']) - 2
    radius = [[] for _ in range(finalTheta)]
    density = [[] for _ in range(finalTheta)]
    temperature = [[] for _ in range(finalTheta)]
    temperatureNew = [[] for _ in range(finalTheta)]
    velocity = [[] for _ in range(finalTheta)]
    firsts = [0] * finalTheta
    thetas = dataFile['x2f'][0:finalTheta]
    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    finalTime = 90000
    indexPlotting = [30, 70, 180]
    for i in range(len(files)):
        dataFile = athena_read.athdf(files[i])
        for j in indexPlotting:
            densityInitial = dataFile['rho'][0][j][finalIndex]
            initialPressure=dataFile['press'][0][j][finalIndex]
            initialTemp=getTemperature(initialPressure,densityInitial)
            if densityInitial < .000001 and i < 30:
                pass
            else:
                if firsts[j] < 3:
                    firsts[j] = firsts[j] + 1
                else:
                    velocityCalc = dataFile['vel1'][0][j][finalIndex]
                    radiusCalc = dataFile['x1f'][finalIndex] + (finalTime - i * 10-t0) * velocityCalc
                    densityCalc = densityInitial * (dataFile['x1f'][finalIndex] / radiusCalc) ** 3
                    temperatureCalc=evolveTemp(dataFile['press'][0][j][finalIndex],densityCalc,densityInitial)
                    #temperatureCalcNew= tempRK4Integrator(tDot,initialTemp,finalTime,50000,t0+i*10,epsilon0,tau,densityInitial)
                    temperatureCalcNew=evolveTempNuclearDecay(initialPressure,densityInitial,epsilon0,tau,t0+i*10,finalTime,50000)
                    radius[j].append(radiusCalc)
                    velocity[j].append(velocityCalc)
                    density[j].append(densityCalc)
                    temperatureNew[j].append(temperatureCalcNew)
                    temperature[j].append(temperatureCalc)
    print("Mass Value:" + str(massIntegration(density, radius, thetas)))
    ###########Plotting###########
    plt.figure()
    c = 3 * (10 ** 10)
    plt.loglog()
    indexPlotting = [30, 70, 180]
    colorStore = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j in range(len(indexPlotting)):
        i = indexPlotting[j]
        thetaValue = thetas[i]

        plt.subplot(211)
        plt.loglog()
        plt.plot(radius[i], density[i], label=f"Theta: {thetaValue:.3f}", color=colorStore[j])
        plt.xlabel("Radius (cm)")
        plt.ylabel("Density (g/cm^3)")
        plt.title(f"Density vs Radius at t={finalTime}")
        plt.legend()
        plt.subplot(212)
        plt.loglog()
        #plt.plot(radius[i], temperature[i], label=f"Theta: {thetaValue:.3f}", color=colorStore[j])
        plt.plot(radius[i], temperatureNew[i], label=f"With Radiation, Theta: {thetaValue:.3f}", color=colorStore[j])
        plt.xlabel("Radius (cm)")
        plt.ylabel("Temperature (K)")
        plt.title(f"Temperature vs Radius at t={finalTime} X=.02")
        plt.legend(fontsize='x-small')
        '''
        plt.subplot(313)
        plt.yscale('log')
        plt.plot(radius[i], c / np.asarray(velocity[i]), label=f'c/v, Theta:{thetaValue:.2f}', color=colorStore[j],
                 linestyle='--')
        plt.plot(radius[i], c / (4 * np.asarray(velocity[i])), color=colorStore[j], linestyle='dotted')
        tauValues = tauIntegration(density[i], radius[i])
        plt.plot(list(reversed(radius[i][0:len(radius[i]) - 1])), tauValues, label=f'Tau, Theta:{thetaValue:.2f}',
                 color=colorStore[j])
        plt.xlabel("Radius")
        plt.title(f"Optical Thickness and c/v vs Radius for various theta (t={finalTime}s)")
        plt.legend(fontsize='x-small')
        '''
    plt.tight_layout()
    plt.legend()
    plt.show()

def compareEvolveRadioactive(files, tau, epsilon0):
    initialTimes=[31,60,90]
    colors=["tab:blue","tab:orange","tab:green"]
    finalTime=100000
    theta=20
    dataFile=athena_read.athdf(files[0])
    thetaVal=dataFile['x2f'][theta]
    count=0
    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    for i in initialTimes:
        dataFile=athena_read.athdf(files[i])
        finalTheta = len(dataFile['x2f']) - 2
        finalIndex = len(dataFile['x1f']) - 2
        densityInitial = dataFile['rho'][0][theta][finalIndex]
        velocityCalc = dataFile['vel1'][0][theta][finalIndex]
        initialTemp = getTemperature(dataFile['press'][0][theta][finalIndex], densityInitial)
        times=np.linspace(i*10,finalTime,50000)
        tempDataAdiabatic=[]
        tempDataRad=[]
        tempDataGas=[]
        time=i*10+t0
        densityData=[]
        tempDataRadioactiveIntegrated = tempRK4Integrator(tDot, initialTemp, finalTime, len(times)-1, t0 + time, epsilon0, tau, densityInitial,verbose=True)
        for j in times:
            radiusCalc = dataFile['x1f'][finalIndex] + (j - i * 10) * velocityCalc
            densityCalc=densityInitial*(dataFile['x1f'][finalIndex]/radiusCalc)**3
            densityData.append(densityCalc)
            temperatureCalc = evolveTemp(dataFile['press'][0][theta][finalIndex], densityCalc, densityInitial)
            tempCalcRad=initialTemp*((i*10)/j)
            tempDataAdiabatic.append(temperatureCalc)
            tempDataRad.append(tempCalcRad)
            tempDataGas.append(initialTemp*(((i*10)/j)**2))
        plt.plot(densityData,tempDataAdiabatic,label=f"Root Find Evolve, Time={time:.2f}", color=colors[count], linewidth=1)
        plt.plot(densityData, tempDataRad,label=f"Radiation Dominated Evolve, Time={time:.2f}", color=colors[count], linestyle='--', linewidth=1)
        #plt.plot(densityData, tempDataGas, label=f"Gas Dominated Evolve, Time={time:.2f}", color=colors[count],linestyle='dotted',linewidth=1)
        plt.plot(densityData,tempDataRadioactiveIntegrated,label=f"Radioactive Integration Method, Time={time:.2f}",color=colors[count],linestyle='dashdot', linewidth=1)
        count=count+1
    plt.loglog()
    plt.ylabel("Temperature")
    plt.xlabel("Density")
    plt.title(f"Temperature vs Density for different evolution methods and Starting Times, Theta={thetaVal:.2f}, Mass Fraction=.1")
    plt.legend()
    plt.show()

def compareMassFractions(files):
    initialTimes = 33
    massFractions=[.02,.05,.1]
    tau=8.77*86400
    colors = ["tab:blue", "tab:orange", "tab:green"]
    finalTime = 100000
    theta = 20
    dataFile = athena_read.athdf(files[0])
    thetaVal = dataFile['x2f'][theta]
    count = 0
    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    dataFile = athena_read.athdf(files[initialTimes])
    finalTheta = len(dataFile['x2f']) - 2
    finalIndex = len(dataFile['x1f']) - 2
    densityInitial = dataFile['rho'][0][theta][finalIndex]
    pressureInitial=dataFile['rho'][0][theta][finalIndex]
    velocityCalc = dataFile['vel1'][0][theta][finalIndex]
    initialTemp = getTemperature(dataFile['press'][0][theta][finalIndex], densityInitial)
    times = np.linspace(t0+initialTimes * 10, finalTime, 50000)
    tempDataAdiabatic = []
    time = initialTimes * 10 + t0
    densityData = []
    for j in times:
        radiusCalc = dataFile['x1f'][finalIndex] + (j - initialTimes * 10) * velocityCalc
        densityCalc = densityInitial * (dataFile['x1f'][finalIndex] / radiusCalc) ** 3
        densityData.append(densityCalc)
        temperatureCalc = evolveTemp(dataFile['press'][0][theta][finalIndex], densityCalc, densityInitial)
        tempDataAdiabatic.append(temperatureCalc)
    plt.plot(densityData, tempDataAdiabatic, label=f"Root Find Evolve, Time={time:.2f}", color="tab:purple",
             linewidth=1)
    for i in massFractions:
        epsilon0=i*1.74*1.6*(10**-6)/(mProton*56*tau)
        tempDataRadioactiveIntegrated = evolveTempNuclearDecay(pressureInitial,densityInitial,epsilon0,tau,t0+i*10,finalTime,50000)
        plt.plot(densityData, tempDataRadioactiveIntegrated, label=f"Radioactive Integration Method, Time={time:.2f}, MFraction={i}",
                 color=colors[count], linewidth=1)
        count = count + 1
    plt.loglog()
    plt.ylabel("Temperature")
    plt.xlabel("Density")
    plt.title(
        f"Temperature vs Density for different evolution methods and Starting Times, Theta={thetaVal:.2f}")
    plt.legend()
    plt.show()

def fullEvolutionCompareRadioactivity(files, tau,epsilon0, massFraction,verbose=False):
    ###############Calculates the final state of the material that leaves the simulation##############
    dataFile = athena_read.athdf(files[0])
    finalIndex = len(dataFile['x1f']) - 2
    finalTheta = len(dataFile['x2f']) - 2
    radius = [[] for _ in range(finalTheta)]
    density = [[] for _ in range(finalTheta)]
    temperatureIntegrator = [[] for _ in range(finalTheta)]
    temperatureNew = [[] for _ in range(finalTheta)]
    velocity = [[] for _ in range(finalTheta)]
    firsts = [0] * finalTheta
    thetas = dataFile['x2f'][0:finalTheta]
    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    finalTime = 90000
    indexPlotting = [30, 70, 180]
    for i in range(len(files)):
        dataFile = athena_read.athdf(files[i])
        for j in indexPlotting:
            densityInitial = dataFile['rho'][0][j][finalIndex]
            initialPressure = dataFile['press'][0][j][finalIndex]
            initialTemp = getTemperature(initialPressure, densityInitial)
            if densityInitial < .000001 and i < 30:
                pass
            else:
                if firsts[j] < 3:
                    firsts[j] = firsts[j] + 1
                else:
                    velocityCalc = dataFile['vel1'][0][j][finalIndex]
                    radiusCalc = dataFile['x1f'][finalIndex] + (finalTime - i * 10 - t0) * velocityCalc
                    densityCalc = densityInitial * (dataFile['x1f'][finalIndex] / radiusCalc) ** 3
                    temperatureCalcNew =evolveTempNuclearDecay(initialPressure,densityInitial,epsilon0,tau,t0+i*10,finalTime,10000)
                    temperatureCalcIntegrator=scipy.integrate.solve_ivp(tDotNew,[i*10+t0,finalTime],[initialTemp],args=(epsilon0,tau,i*10+t0,densityInitial),max_step=(finalTime-i*10-t0)/100)['y'][0][-1]
                    radius[j].append(radiusCalc)
                    velocity[j].append(velocityCalc)
                    density[j].append(densityCalc)
                    temperatureNew[j].append(temperatureCalcNew)
                    temperatureIntegrator[j].append(temperatureCalcIntegrator)
        print("SHIMMYIMMY AHH:"+str(i))
    print("Mass Value:" + str(massIntegration(density, radius, thetas)))
    ###########Plotting###########
    plt.figure()
    c = 3 * (10 ** 10)
    plt.loglog()
    if verbose:
        indexPlotting = [30, 70, 180]
        colorStore = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for j in range(len(indexPlotting)):
            i = indexPlotting[j]
            thetaValue = thetas[i]
            plt.loglog()
            error=np.abs(np.asarray(temperatureNew[i])**4-np.asarray(temperatureIntegrator[i])**4)/np.asarray(temperatureIntegrator[i])**4
            plt.plot(radius[i], error, label=f"Relative Error Between Methods Theta: {thetaValue:.3f}",
                     color=colorStore[j])
            plt.xlabel("Radius (cm)")
            plt.ylabel("Relative Error")
            plt.title(f"Relative Error vs Radius at t={finalTime}, Mass Fraction={massFraction}")
            # plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.legend()
        plt.show()
    indexPlotting = [30, 70, 180]
    colorStore = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j in range(len(indexPlotting)):
        i = indexPlotting[j]
        thetaValue = thetas[i]
        plt.loglog()
        plt.plot(radius[i], np.asarray(temperatureNew[i])**4, label=f"Step Method Theta: {thetaValue:.3f}", color=colorStore[j])
        plt.plot(radius[i],np.asarray(temperatureIntegrator[i])**4,label=f"Integrator Method, Theta: {thetaValue:.3f}", color=colorStore[j], linestyle='--')
        plt.xlabel("Radius (cm)")
        plt.ylabel("Temperature^4 (K)")
        plt.title(f"Temperature^4 vs Radius at t={finalTime}, Mass Fraction={massFraction}")
        #plt.legend(fontsize='x-small')
    plt.tight_layout()
    plt.legend()
    plt.show()

def evolveWithRad(files,tau,epsilon0,finalTime,passedIndices=[]):
    dataFile = athena_read.athdf(files[0])
    finalIndex = len(dataFile['x1f']) - 2
    finalTheta = len(dataFile['x2f']) - 2
    radius = [[] for _ in range(finalTheta)]
    density = [[] for _ in range(finalTheta)]
    temperatureIntegrator = [[] for _ in range(finalTheta)]
    temperatureNew = [[] for _ in range(finalTheta)]
    velocity = [[] for _ in range(finalTheta)]
    firsts = [0] * finalTheta
    indexPlotted=[]
    thetas=dataFile['x2f'][0:finalTheta]
    if passedIndices==[]:
        indexPlotted=np.linspace(0,len(thetas)-1,len(thetas))
    else:
        indexPlotted=passedIndices[0:]

    t0 = dataFile['x1f'][0] / (2.53 * (10 ** 9))
    #indexPlotting = [30, 70, 180]
    for i in range(len(files)):
        dataFile = athena_read.athdf(files[i])
        for j in indexPlotted:
            densityInitial = dataFile['rho'][0][j][finalIndex]
            initialPressure = dataFile['press'][0][j][finalIndex]
            initialTemp = getTemperature(initialPressure, densityInitial)
            if densityInitial < .000001 and i < 30:
                pass
            else:
                if firsts[j] < 3:
                    firsts[j] = firsts[j] + 1
                else:
                    velocityCalc = dataFile['vel1'][0][j][finalIndex]
                    radiusCalc = dataFile['x1f'][finalIndex] + (finalTime - i * 10 - t0) * velocityCalc
                    densityCalc = densityInitial * (dataFile['x1f'][finalIndex] / radiusCalc) ** 3
                    temperatureCalcIntegrator = evolveTempNuclearDecay(initialPressure,densityInitial,epsilon0,tau,t0+i*10,finalTime,1000)
                    radius[j].append(radiusCalc)
                    velocity[j].append(velocityCalc)
                    density[j].append(densityCalc)
                    temperatureIntegrator[j].append(temperatureCalcIntegrator)
        print("SHIMMYIMMY AHH:" + str(i))
    return finalIndex, finalTheta, radius, density, temperatureIntegrator, velocity, t0, thetas

#gives you a profile of all the material at an early time that you can then do what you want with
def evolveMaterialEarly(files):
    finalIndex, finalTheta, radius, density, temperature, velocity, t0, thetas = evolveNoRad(files, 1000)
    t0=1000+t0

    #get mass along each dtheta wide ray so that when we integrate in we have a mass fraction along the ray
    totalMass=[]
    dTheta=thetas[1]-thetas[0]
    massValue=0
    for i in range(len(density)-1):
        for j in range(len(density[i])-1):
            integrationChunk = density[i][j] * (radius[i][j] ** 2) * np.sin(thetas[i+1]) * (abs(radius[i][j + 1] - radius[i][j])) * dTheta
            massValue = massValue + integrationChunk
        totalMass.append(massValue)
    totalMass.append(totalMass[-1])
    finalTime=150000
    startTime=10000
    stepSize=1000
    steps=int((finalTime-startTime)/stepSize)
    times=np.linspace(startTime,finalTime,steps)
    densityTemp=density[0:]
    radiusTemp=radius[0:]
    velocityTemp=velocity[0:]
    done=[False]*len(density)
    temperatureRadiating=[0]*len(density)
    radiusRadiating=[0]*len(density)
    #Layout:
    for time in times:
        for i in range(len(densityTemp)):
            for j in range(len(densityTemp[i])):
                radiusCalc=radiusTemp[i][j]+(time-t0)*velocityTemp[i][j]
                densityCalc=density[i][j]*((radiusTemp[i][j]/radiusCalc)**3)
                radiusTemp[i][j]=radiusTemp[i][j]+(time-t0)*velocityTemp[i][j]
                densityTemp[i][j]=densityCalc
        for i in range(len(densityTemp)-1):
            c=3*(10**8)
            tauValue=0
            densityStore=0
            for j in range(len(densityTemp[i])):
                integrationChunk = .2 * (densityTemp[i][-j] + .5 * (
                densityTemp[i][-j] - density[i][-j]))* abs(radius[i][-j-1] - radius[i][-j])
                tauValue = tauValue + integrationChunk
                densityStore=densityStore+integrationChunk*(radius[i][j] ** 2) * np.sin(thetas[i+1]) *dTheta/.2
                if tauValue>(c/velocityTemp[i][j]):
                    break
            if j==0:
                break
            elif (not done[i]) and densityStore / totalMass[i] > 10 ** -2:
                done[i] = True
                print("Temp Evolution:")
                print(temperature[i][-j])
                print(density[i][-j])
                print(pressureEq(temperature[i][-j],density[i][-j]))
                print("ELINE")
                radiusRadiating[i] = radiusTemp[i][-j]
                temperatureRadiating[i] = evolveTemp(pressureEq(temperature[i][-j],density[i][-j]),densityTemp[i][-j],density[i][-j])
    plt.figure()
    plt.subplot(211)
    plt.yscale('log')
    plt.plot(thetas, temperatureRadiating)
    plt.title("Temperature that 1e-2 mass coordinate Radiates at vs Angle")
    plt.ylabel("Radiating Temperature (K)")
    plt.xlabel("Theta (rad)")
    plt.subplot(212)
    plt.yscale('log')
    plt.plot(thetas, radiusRadiating)
    plt.title("Radius that 1e-2 mass coordinate Radiates at vs Angle")
    plt.ylabel("Radiating Radius (cm)")
    plt.xlabel("Theta (rad)")
    plt.tight_layout()
    plt.show()
                #this is the background optical depth integration.
    #step that updates the densityTemp, radiusTemp, and velocityTemp arrays like in actual evolveEdge function
    #step that integrates inwards until optical thickness>c/v

#gives you a profile of all the material at an early time that you can then do what you want with
def evolveMaterialEarlyWithRadiation(files,epsilon0,tau):
    finalIndex, finalTheta, radius, density, temperature, velocity, t0, thetas = evolveNoRad(files, 1000)
    t0=1000+t0

    #get mass along each dtheta wide ray so that when we integrate in we have a mass fraction along the ray
    totalMass=[]
    dTheta=thetas[1]-thetas[0]
    massValue=0
    for i in range(len(density)-1):
        for j in range(len(density[i])-1):
            integrationChunk = density[i][j] * (radius[i][j] ** 2) * np.sin(thetas[i+1]) * (abs(radius[i][j + 1] - radius[i][j])) * dTheta
            massValue = massValue + integrationChunk
        totalMass.append(massValue)
    totalMass.append(totalMass[-1])
    finalTime=150000
    startTime=10000
    stepSize=1000
    steps=int((finalTime-startTime)/stepSize)
    times=np.linspace(startTime,finalTime,steps)
    densityTemp=density[0:]
    radiusTemp=radius[0:]
    velocityTemp=velocity[0:]
    done=[False]*len(density)
    temperatureRadiating=[0]*len(density)
    radiusRadiating=[0]*len(density)
    #Layout:
    for time in times:
        for i in range(len(densityTemp)):
            for j in range(len(densityTemp[i])):
                radiusCalc=radiusTemp[i][j]+(time-t0)*velocityTemp[i][j]
                densityCalc=density[i][j]*((radiusTemp[i][j]/radiusCalc)**3)
                radiusTemp[i][j]=radiusTemp[i][j]+(time-t0)*velocityTemp[i][j]
                densityTemp[i][j]=densityCalc
        for i in range(len(densityTemp)-1):
            c=3*(10**8)
            tauValue=0
            densityStore=0
            for j in range(len(densityTemp[i])):
                integrationChunk = .2 * (densityTemp[i][-j] + .5 * (
                densityTemp[i][-j] - density[i][-j]))* abs(radius[i][-j-1] - radius[i][-j])
                tauValue = tauValue + integrationChunk
                densityStore=densityStore+integrationChunk*(radius[i][-j] ** 2) * np.sin(thetas[i+1]) *dTheta/.2
                if tauValue>(c/velocityTemp[i][-j]):
                    break
            if j==0:
                break
            elif (not done[i]) and densityStore/totalMass[i]>10**-2:
                done[i]=True
                radiusRadiating[i]=radiusTemp[i][-j]
                print(radiusRadiating)
                print(tauValue)
                print(densityStore)
                print(totalMass[i])
                temperatureRadiating[i]=evolveTempNuclearDecay(pressureEq(temperature[i][-j],density[i][-j]),density[i][-j],epsilon0,tau,t0,time,10000)
    plt.figure()
    plt.subplot(211)
    plt.yscale('log')
    plt.plot(thetas,temperatureRadiating)
    plt.title("Temperature that 1e-2 mass coordinate Radiates at vs Angle with Radioactivity")
    plt.ylabel("Radiating Temperature (K)")
    plt.xlabel("Theta (rad)")
    plt.subplot(212)
    plt.yscale('log')
    plt.plot(thetas,radiusRadiating)
    plt.title("Radius that 1e-2 mass coordinate Radiates at vs Angle")
    plt.ylabel("Radiating Radius (cm)")
    plt.xlabel("Theta (rad)")
    plt.tight_layout()
    plt.show()
                #this is the background optical depth integration.
    #step that updates the densityTemp, radiusTemp, and velocityTemp arrays like in actual evolveEdge function
    #step that integrates inwards until optical thickness>c/v
#returns derivative at midpoints
def calcDerivative(xData,yData):
    derivative=[]
    for i in range(len(yData)-1):
        derivative.append((yData[i+1]-yData[i])/np.abs((xData[i+1]-xData[i])))
    return derivative
#takes temperature, density, and radius for a given angle and returns flux as a function of radius in format: radii, fluxes
def calcFluxData(temperature, density, radius,verbose=False,theta=""):
    numpyTemp=np.asarray(temperature)
    numpyDensity=np.asarray(density)
    numpyRadius=np.asarray(radius)
    prefixArray=(4*a*(numpyTemp**3)/(.2*numpyDensity))
    tempDerivative=calcDerivative(numpyRadius,numpyTemp)
    prefixArrayAveraged=[]
    newXValues=[]
    for i in range(len(prefixArray)-1):
        prefixArrayAveraged.append((prefixArray[i+1]+prefixArray[i])/2.0)
        newXValues.append((numpyRadius[i+1]+numpyRadius[i])/2.0)
    fluxArray=np.asarray(prefixArrayAveraged)*tempDerivative
    finalDerivative=calcDerivative(newXValues,fluxArray)
    finalXVals=numpyRadius[1:len(numpyRadius)-1]
    if verbose:
        plt.subplot(211)
        plt.plot(radius, temperature)
        plt.ylabel("Temperature")
        plt.xlabel("Radius")
        plt.title("Temperature vs Radius "+theta)
        plt.subplot(212)
        plt.plot(newXValues, fluxArray)#savgol_filter(fluxArray,20,2))
        plt.ylabel("Flux/c")
        plt.xlabel("Radius")
        plt.title("Flux/c vs Radius "+theta)
        plt.tight_layout()
        plt.show()
    return finalXVals, finalDerivative#/numpyDensity[1:len(numpyDensity)-1]

#this function plots the divergence of the radiative flux as a function of radius (ignoring angular components)
def plotFluxes(files,tau,epsilon0):
    finalTime=90000
    indexPlotting=[30,70,180]
    thetaStore=athena_read.athdf(files[0])['x2f']
    thetaPassed=[]
    for index in indexPlotting:
        thetaPassed.append(thetaStore[index])
    finalIndex=0
    finalTheta=0
    radius=[]
    density=[]
    temperature=[]
    velocity=[]
    t0=0
    thetas=[]
    if epsilon0==0.0:
        finalIndex, finalTheta, radius, density, temperature, velocity, t0, thetas=evolveNoRad(files,finalTime)
    else:
        finalIndex, finalTheta, radius, density, temperature, velocity, t0, thetas = evolveWithRad(files,tau,epsilon0,finalTime,passedIndices=indexPlotting)
    for i in indexPlotting:
        c=(3*(10**8))
        thetaVal=thetaStore[i]
        fluxRadii, fluxVals= calcFluxData(temperature[i],density[i],radius[i],verbose=False,theta=f"Theta:{thetaVal:.3f}")
        print(fluxRadii)
        print(fluxVals)
        thetaVal=thetaStore[i]
        thetaString=f"Theta={thetaVal:.3f}"
        fluxModified=(finalTime/3.0)*np.abs(fluxVals)*c/((4.0/3.0)*(a*(np.asarray(temperature[i][0:len(fluxVals)])**4)))
        plt.plot(fluxRadii,savgol_filter(fluxModified,20,1),label=thetaString)
    plt.xlabel("Radius (cm)")
    plt.ylabel("|dif(F)/(rho)|  (only doing F_r)")
    plt.title("|dif(F)/(rho)| vs Radius for Various Angles")
    plt.legend()
    plt.loglog()
    plt.show()

def massIntegrationInwards(file):
    data=athena_read.athdf(file)
    density=data['rho']
    radius=data['x1f']
    thetas=data['x2f']

    mArray=[]
    radiusStore=[]
    M=0
    for i in range(int(len(radius)/2.0)):
        for j in range(len(thetas)-1):
            M+= density[0][j][-(i+1)]*(radius[-(i+1)]**2)*2*np.pi*(radius[-(i+1)]-radius[-i-2])*np.sin(thetas[j])*(thetas[j+1]-thetas[j])/(1.789623*(10.0**33))
        radiusStore.append(radius[-(i+1)])
        mArray.append(M)
    print(len(mArray))
    for i in range(len(mArray)):
        if mArray[i]>.5:
            print(i)
            print(radius[-i-1])
            break
    print(mArray[-1])
    plt.plot(radiusStore,mArray)
    plt.xlabel("Radius (cm)")
    plt.ylabel("Mass Integrated In/Total Mass")
    plt.title("Mass fraction vs radius for integrating inwards")
    plt.show()

def analyzeShockEntropy(files):
    entropyJump=[]
    for file in files[1:11]:
        data=athena_read.athdf(file)
        densityData=data['rho'][0][0]
        pressureData=data['press'][0][0]
        temperature=getTemperature(pressureData,densityData)
        entropy=getEntropy(temperature, densityData)
        planetIndex=0
        for j in range(len(densityData)):
            if densityData[j]<10**-5:
                planetIndex=j
                break
        entropyJumpVal=entropy[planetIndex-3]-entropy[0]
        entropyJump.append(entropyJumpVal)
    x=np.linspace(10,110,10)
    plt.plot(x,entropyJump)
    plt.xlabel("Time")
    plt.ylabel("Entering Simulation vs Post Shock Entropy Jump")
    plt.title("Entropy Jump across Shock vs Time (very rough)")
    plt.show()

@njit
def getForces(forceArray, phi):
    currentForce=0
    totalMomentum=0
    for i in range(1000):
        for j in range(100):
            for k in range(len(phi) - 1):
                forceStore = forceArray[k][j][i]
                currentForce = currentForce + forceStore
                totalMomentum = (totalMomentum + forceStore)*10
    return currentForce, totalMomentum

def calcForcesAtEveryTime(files):
    forces=[]
    totalMomentum=0
    currentForce=0
    for l in range(len(files)):
        dataFile=athena_read.athdf(files[l])
        currentForce = 0
        forceArray=dataFile['user_out_var0']
        phi=dataFile['x3f']
        currentForce, tempMomentum=getForces(forceArray,phi)
        totalMomentum=totalMomentum+tempMomentum
        forces.append(currentForce)
        print("OK "+str(l))
    print(totalMomentum)
    times=np.linspace(0,1000,len(files))
    plt.plot(times[1:], forces[1:])
    plt.xlabel("Time (units of 10s)")
    plt.ylabel("Force")
    plt.yscale('log')
    plt.title("Net Force on Star over Time")
    plt.xlim((0,1000))
    plt.show()

def testRectangularPlot(data1):
    thetas=data1['x2f']
    radiusArray=data1['x1f']
    density = data1['rho'][0]
    pressure = data1['press'][0]
    temperature = getTemperature(pressure, density)
    x=[]
    y=[]
    for r in radiusArray:
        for theta in thetas:
            x.append(r*np.cos(theta))
            y.append(r*np.sin(theta))

def flipData(thetas, temperature, density):
    thetaNew=np.concatenate((-1.0*thetas, thetas[1:]))
    temperatureNew=temperature[1:]

#find the modes for the first n spherical harmonics assuming m=0
def sphericalHarmonicDecomposition(data,thetas, n):
    endTheta=thetas[-1]
    resRate=len(thetas)/endTheta
    extraThetas=np.linspace(endTheta, np.pi, int(resRate*(np.pi-endTheta)))
    extraData=np.zeros(len(extraThetas))+data[-1]
    fullTheta=np.concatenate((thetas, extraThetas))
    fullData=np.concatenate((data,extraData))
    spectrumCoefficients=[]
    for i in range(n):
        tempData=fullData*sph_harm(0,i,0, fullTheta)*np.sin(fullTheta)
        tempData=tempData.real
        tempValue=integrateData1D(tempData,fullTheta)
        spectrumCoefficients.append(np.abs(2*np.pi*tempValue))
    return spectrumCoefficients

def makeThetaArray(data, thetas, normalize=True):
    angleArray=[]
    for i in range(len(thetas)):
        angleArray.append(data[0][i][-1])
    if normalize:
        angleArray=np.asarray(angleArray)
        angleArray=angleArray/np.max(angleArray)
    return np.asarray(angleArray)

def plotHarmonicsDensity(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['rho'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)
        #print(coefficients)
        plt.plot(coefficients, label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Density")
    plt.show()

def plotHarmonicsPressure(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['press'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)
        #print(coefficients)
        plt.plot(coefficients, label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Pressure")
    plt.show()

def plotHarmonicsMomentum(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['rho']*data['vel1'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)
        #print(coefficients)
        plt.plot(coefficients, label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Radial Momentum")
    plt.show()

def plotHarmonicsDensitySmooth(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['rho'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)

        #print(coefficients)
        plt.plot(savgol_filter(coefficients,20,1), label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Density")
    plt.show()

def plotHarmonicsPressureSmooth(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['press'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)

        #print(coefficients)
        plt.plot(savgol_filter(coefficients,20,1), label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Pressure")
    plt.show()

def plotHarmonicsMomentumSmooth(files, times):
    data=athena_read.athdf(files[0])
    thetas=data['x2f'][0:len(data['x2f'])-1]
    for time in times:
        data=athena_read.athdf(files[time])
        density=makeThetaArray(data['rho']*data['vel1'],thetas)
        coefficients=sphericalHarmonicDecomposition(density, thetas, 100)

        #print(coefficients)
        plt.plot(savgol_filter(coefficients,20,1), label=f'Time:{time*10}')
    plt.xlabel("l")
    plt.loglog()
    plt.ylabel("a_lm")
    plt.legend()
    plt.title("Spherical Harmonic Expansion Coefficients of Momentum")
    plt.show()







