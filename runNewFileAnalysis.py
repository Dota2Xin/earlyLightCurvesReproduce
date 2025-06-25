from newFileAnalysis import *
import time

def main():
    plt.style.use("./mesa.mplstyle")
    innerRadiusValue=3*(10**11)
    #files=getFiles()
    data=getDataDict("supernovaCollisionFinalRun")

    #radius, density, velocity= getSNRData(data,files[-1], innerRadiusValue, 7.5e11, .9e10)
    #processSNRData(radius, density, velocity, "updatedSNRData")
    #radius, density, velocity= getDynamicData(data,.9e10,7.5e11,3000, verbose=True)
    #density, velocity, temperature, radius, thetas=getDynamicData(data,.9e10,7.5e11,3000, verbose=False, interpolate=True)
    #makeHDF5data(density, velocity, radius, thetas, "test1")
    #readHDF5data("test1")

    ########NICKEL HEATING###############
    '''
    massFraction = .16
    tau = 757728
    epsilon0 = massFraction * (1.72 * (1.6 * (10 ** -6)) / (56 * tau * mProton))

    #processEntropyGeneration(data, .9e10, 7.5e11, epsilon0/massFraction, tau)

    #print(mProton)
    #for two hours, 7200
    interpDensity, interpPressure,interpVelocity,gridRadii, thetas=evolveEdgeFull(data, .9e10,7.5e11,7200,epsilon0, tau)
    makeHDF5data(interpDensity,interpVelocity, gridRadii,interpPressure, thetas, "finalInputDataSixteenPercent")
    time.sleep(3)
    #readHDF5data("fullDataFourPercent4_4_25")

    '''
    '''
    pressure=getHDF5data("fullData1_15_24", "pressure")
    density = getHDF5data("fullData1_15_24", "density")
    radii=getHDF5data("fullData1_15_24", "radius")
    thetas=getHDF5data("fullData1_15_24", "theta")
    velocity=getHDF5data("fullData1_15_24", "velocity")
    plotFinalSnapshot(pressure, radii, thetas, "pressure", 1e3, 1e7)
    plotFinalSnapshot(density, radii, thetas, "density", 1e-11, 1e-5)


    temperature = np.copy(pressure)
    for i in range(len(pressure)):
        for j in range(len(pressure[0])):
            temperature[i][j]=getTemperature(pressure[i][j],density[i][j])
    plotFinalSnapshot(temperature, radii, thetas, "Temperature", 5e3, 5e5)
    plotFinalSnapshot(velocity, radii, thetas, "Velocity", 5e8, 5e9)
    '''


    #plotFinalSnapshot(velocity, radii, thetas, "Velocity", 5e8, 5e9)
    directory="fullDataFourPercent4_4_25"
    pressure = getHDF5data( directory, "pressure")
    density = getHDF5data( directory, "density")
    radii = getHDF5data( directory, "radius")
    thetas = getHDF5data( directory, "theta")
    velocity = getHDF5data( directory, "velocity")
    print(len(pressure))
    radiusValue=(1.5*(10**13.0))
    #plotRadRatio(radii,radiusValue,density,pressure,thetas)
    print(radii)
    print(density[0][100])
    print(pressure[0][100])
    print(radii[100])
    temperature = np.copy(pressure)
    for i in range(len(pressure)):
        for j in range(len(pressure[0])):
            temperature[i][j] = getTemperature(pressure[i][j], density[i][j])
    radPressure=a*np.power(temperature,4.0)/3.0
    gasPressure=density*temperature*kB/(mu*mProton)
    #plotFinalSnapshot(radPressure/gasPressure, radii, thetas, "Rad Pressure/Gas Pressure", 1e3, 1e-3, "")
    storeMinDensity = np.min(density)
    storeMinTemperature = np.min(temperature)
    for i in range(len(density)):
        for j in range(len(density[0])):
            if density[i][j] == density[-1][-1]:
                density[i][j] = 0
            if temperature[i][j] == storeMinTemperature:
                temperature[i][j] = 0

    # Logan's Code:
    myrlim = 390
    mythetalim = 80.0 / 180.0 * np.pi
    mylabelpad = 10

    plotEvolvedTemp(radii, density, temperature, velocity, thetas)
    mynorm = matplotlib.colors.LogNorm(vmin=1e4, vmax=3e5)
    fig = plt.figure()
    plt.polar()
    rSun = 6.956 * (10 ** 10.0)
    im = plt.pcolormesh(thetas, radii / (rSun), np.transpose(temperature)[:-1,:-1], cmap='inferno',
                        norm=mynorm, shading='flat')
    cbar = plt.colorbar(im)
    cbar.set_label(r'$T$(K)', fontsize=10)
    cbar.ax.tick_params(labelsize=14)
    ax = plt.gca()
    ax.set_rlim(0.0, myrlim)
    ax.set_thetalim(0.0, mythetalim)
    plt.xticks(np.array([20.0, 40.0, 60.0, 80.0]) * np.pi / 180.0)
    ax.tick_params(labelsize=10)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    plt.xlabel(r'$r$ ($\rm{R}_{\odot}$)', fontsize=10, labelpad=mylabelpad)
    plt.annotate(r'$t=2$ hrs', xy=(1.0, 1.0), xytext=(.63, 0.85), xycoords='figure fraction', color='k', size=12)
    plt.annotate(r'$\rm{X}_{56}=\rm{0.04}$', xy=(1.0, 1.0), xytext=(.63, 0.75), xycoords='figure fraction', color='k', size=12)
    plt.tight_layout()
    plt.show()

    #plotFinalSnapshotDouble(density,temperature,  radii, thetas, "Density", 1e-11, 1e-5, "$\\textrm{g/cm}^3$ or K/$10^{12}$")

    #processHDF5("finalInputDataSixteenPercent", "fullDataSixteenPercent")
    #pressure = getHDF5data("fullData4_26_24", "pressure")
    #density = getHDF5data("fullData4_26_24", "density")
    #radii = getHDF5data("fullData4_26_24", "radius")
    #thetas = getHDF5data("fullData4_26_24", "theta")
    #velocity = getHDF5data("fullData4_26_24", "velocity")
    thetaIndex=[10,80,197]
    #plotRadiusGraphDensity(thetas,thetaIndex,radii,density)
    #plotRadiusGraphTemperature(thetas,thetaIndex,radii,density,pressure)
    #plotRadiusGraphRadRatio(thetas,thetaIndex,radii,density,pressure)


if __name__=="__main__":
    main()