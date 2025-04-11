import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from copy import deepcopy

def standardPlot(data,ax=None,ls="-"):
    if ax is None:
        fig,ax = plt.subplots(len(data.keys())-1,1,figsize=(6,1.8*len(data.keys())),dpi=100,sharex=True)
    time = data["time"]
    for i,(key,val) in enumerate(data.items()):
        if key != "time":
            ax[i].set_title(key)
            if len(val.shape)<3:
                ax[i].plot(time,val,ls=ls)
            else:
                ax[i].plot(time,val[:,-1,3],ls=ls)

def sideBySidePlot(data1,data2,ax=None):
    if ax is None:
        fig,ax = plt.subplots(len(data1.keys())-1,1,figsize=(6,1.8*len(data1.keys())),dpi=100,sharex=True)
    standardPlot(data1,ax,ls="-")
    standardPlot(data2,ax,ls="--")


def sigmoid(z,h,kInv,minOp):
    return minOp +  (1.0-minOp)/(1.0 + np.exp((h-z)*kInv))

def solveGate(x,s,TS,xh,xk,xt0,xt1,xth,xtk,minOpenness):
    xInf = sigmoid(s,xh,xk,minOpenness)#*(xt0+xt1)/TS
    tau = xt0 + xt1*sigmoid(s,xth,xtk,0)
    #return x + TS*(xInf-x)/tau
    return (tau*x + TS*xInf)/(tau+TS)
def calcWeightChange(p,Ca,rate,gammaP,gammaD,thetaP,thetaD):
    #from the Graupner and Brunel paper
    #rate = 1/tau 
    pStar = 0.5
    revP = (1-p)
    fp = -p*revP*(pStar-p)
    potentiation = gammaP*revP*float(Ca>thetaP)
    depression = -gammaD*p*float(thetaD<Ca<=thetaP)
    Noise = 0
    return rate*fp+potentiation+depression+Noise

def solveChannels(ac,spikeOverwriteList,TS):
    nChannels = ac["nChannels"]
    voltage = ac["voltages"]
    calciumVals = ac["calciumVals"]
    writeIndex = ac["writeIndex"]
    maxDelay = ac["maxDelayInt"]

    for i in range(nChannels):
        postSynapticNeuron = ac["postNeuron"][i]
        V = voltage[postSynapticNeuron]
        Ca = calciumVals[postSynapticNeuron]
        preSynapticNeuron = ac["preNeuron"][i] #for channel this is = to postSynaptic neuron, and does not matter
        #spikes = max(spikeVals[preSynapticNeuron],spikeOverwriteList[i])
        xs = ac["x"][i]
        xhs,xks = ac["vHalf"][i],ac["vKi"][i]
        xts0,xts1 = ac["tau0"][i],ac["tau1"][i]
        xths,xtks = ac["th"][i],ac["tkInv"][i]
        
        exponents = ac["exponent"][i]
        minOpenness = ac["minOpenness"][i]
        Erev = ac["Erev"][i]
        g0 = ac["conductance"][i]
        p = ac["efficacy"][i]
        w = ac["conductanceGain"][i]
        CaCurrent = ac["CaCurrent"][i]

        tempTS = TS
        gatingTotal = 1

        for j in range(len(xs)):
            x = xs[j]
            n = exponents[j]
            gatingTotal *= (x**n)

        g = g0 + w*p
        I = g*gatingTotal*(V-Erev) 
        ac["netCurrent"][postSynapticNeuron] += I
        ac["netCaCurrent"][postSynapticNeuron] += CaCurrent*I
        ac["current"][i] = I

        synapseDelay = ac["inputDelay"][i]
        readIndex = int((writeIndex[preSynapticNeuron]-synapseDelay)%maxDelay)
        perceivedSpike = ac["spikeBuffer"][preSynapticNeuron][readIndex]
        perceivedSpike = max(perceivedSpike,spikeOverwriteList[i])

        xs[0] = solveGate(xs[0],V,tempTS,xhs[0],xks[0],xts0[0],xts1[0],xths[0],xtks[0],minOpenness[0]) #gate1
        xs[1] = solveGate(xs[1],V,tempTS,xhs[1],xks[1],xts0[1],xts1[1],xths[1],xtks[1],minOpenness[1]) #gate2
        xs[2] = solveGate(xs[2],Ca,tempTS,xhs[2],xks[2],xts0[2],xts1[2],xths[2],xtks[2],minOpenness[2]) #CaGate
        xs[3] = solveGate(xs[3],perceivedSpike,tempTS,xhs[3],xks[3],xts0[3],xts1[3],xths[3],xtks[3],minOpenness[3]) #spikeGate

        dp = calcWeightChange(p,Ca,ac["rate"][i],ac["gammaP"][i],ac["gammaD"][i],ac["thetaP"][i],ac["thetaD"][i])
        ac["efficacy"][i]+=TS*dp

def solveDiffusiveConnections(ac):
    netCurrent = ac["netCurrent"]
    netCaCurrent = ac["netCaCurrent"]
    nDiffusiveConnections = ac["nDiffusiveConnections"] 
    for k in range(nDiffusiveConnections):
        #we could maybe implement this as a type of channel and use the above code but that seems more messy and might not even be efficient
        postSynapticNeuron = ac["diffPostNeuron"][k]
        Vpost = ac["voltages"][postSynapticNeuron]
        Capost = ac["calciumVals"][postSynapticNeuron]
        preSynapticNeuron = ac["diffPreNeuron"][k]
        Vpre = ac["voltages"][preSynapticNeuron]
        Capre = ac["calciumVals"][preSynapticNeuron]
        diffusionRate = ac["diffRate"][k]
        I = diffusionRate*(Vpost-Vpre)
        CaFlow = diffusionRate*(Capost-Capre) #technically this is electrodiffusion so I should probably implement that correctly 

        netCurrent[postSynapticNeuron] -= I
        netCaCurrent[postSynapticNeuron] += CaFlow*I

        netCurrent[preSynapticNeuron] += I
        netCaCurrent[preSynapticNeuron] -= CaFlow*I

def solveNeurons(ac,TS):
    spikeBuffer = ac["spikeBuffer"]
    voltage = ac["voltages"]
    nNeurons = ac["nNeurons"]
    calciumVals = ac["calciumVals"]
    writeIndex = ac["writeIndex"]
    alpha = 1/(2*96485) #this is the conversion factor between current and Ca diffusion. scale it up if you want Ca in different units 
    CaEquilibriumVal = 0 #1 I think 1e-6 is reasonble, otsuka is 0
    CaRemovalRate = 2 #ms, this is the value otsuka uses, not sure where they get it from 
    for i in range(nNeurons):
        spikeBuffer[i][writeIndex[i]] = int(voltage[i]>0)
        voltage[i] -= TS*(ac["netCurrent"][i]+ac["biasCurrents"][i])
        calciumVals[i] += TS*(-alpha*ac["netCaCurrent"][i] + (-calciumVals[i]+ CaEquilibriumVal)*CaRemovalRate) #This requires some explaination:
                        #We need to convert Ca current (in uA/cm^2) to Ca concentration (in mol/L) so I hardcode the conversion factor 
                        #Cellular Ca is tightly regulated to ~uM concentrations, so I've included a fixed point at 1e-6.
                        #I've set the timeconstant for Ca regulation to be 0.5ms, which might be a little fast
                        #With this implementation its possible to have 1. calcium efflux (which is fine), and 2. negative Ca concentrations (which is not fine)
                        #but negative Ca would require persistent Vm>140mV which is unphysiological anyway. 
        #spikeVals[i] = float(voltage[i] > 0) #we could add delays by making spikes be some dynamical system triggered by voltage > 0. (do a phase portrait to figure it out)            
        writeIndex[i] = (writeIndex[i]+1)%ac["maxDelayInt"]

def solveSystemStandard(ac,TS,runTime,stimInfo={},
                        recordingKeys = ["voltages","calciumVals","netCurrent","current"]):
    nIterations = int(runTime/TS)
    data = {key:[] for key in recordingKeys}
    tv = np.linspace(0,runTime,nIterations)
    nNeurons = ac["nNeurons"]
    nChannels = ac["nChannels"]
    for iteration,t in enumerate(tv):   
        #calculate currents
        ac["netCurrent"] = np.zeros([nNeurons])
        ac["netCaCurrent"] = np.zeros([nNeurons])
        spikeOverwriteList = np.zeros(nChannels)
        if iteration in stimInfo.keys():
            for syn in stimInfo[iteration]:
                spikeOverwriteList[syn] = 1

        #update the gates
        solveChannels(ac,spikeOverwriteList,TS)

        solveDiffusiveConnections(ac)

        solveNeurons(ac,TS)

        for key in data:
            data[key].append(deepcopy(ac[key]))

    for key in data:
        data[key] = np.array(data[key])
    data["time"] = tv
    return data