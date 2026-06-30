import numpy as np
import matplotlib.pyplot as plt
import os,sys
from copy import deepcopy
import tqdm

def standardPlot(data,ax=None,ls="-"):
    if ax is None:
        fig,ax = plt.subplots(len(data.keys())-1,1,figsize=(6,1.8*len(data.keys())),dpi=100,sharex=True)
    time = data["time"]
    if type(ax) is not np.ndarray:
        ax = [ax]
    for i,(key,val) in enumerate(data.items()):
        if key != "time":
            ax[i].set_title(key)
            if len(val.shape)<3:
                ax[i].plot(time,val,ls=ls)
            else:
                ax[i].plot(time,val.sum(axis=2),ls=ls)

def sideBySidePlot(data1,data2,ax=None):
    if ax is None:
        fig,ax = plt.subplots(len(data1.keys())-1,1,figsize=(6,1.8*len(data1.keys())),dpi=100,sharex=True)
    standardPlot(data1,ax,ls="-")
    standardPlot(data2,ax,ls="--")

def gatingPlot(data,ax=None):
    time = data["time"]
    gatingData = data["x"]
    nChannels = gatingData.shape[1]
    nGates = gatingData.shape[2]
    if ax is None:
        fig,ax = plt.subplots(nChannels,nGates,figsize = (12,2.5*nChannels),dpi=120,sharex=True,sharey=True)

    if len(ax.shape) ==1:
        ax = np.array([ax,ax])
    for i in range(nChannels):
        for j in range(nGates):
            ax[i,j].plot(time,gatingData[:,i,j])
            ax[i,j].set_title(f"{i},{j}")

    return ax

def gatingPlotSideBySide(data1,data2):
#fig,ax = plt.subplots(nChannels,nGates,figsize = (12,1.8*nChannels),dpi=120,sharex=True,sharey=True)
    ax = gatingPlot(data1,None)   
    gatingPlot(data2,ax)

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

def calcWeightChangeVectorized(ac,TS):
    rate,gammaP,gammaD,thetaP,thetaD = ac["rate"],ac["gammaP"],ac["gammaD"],ac["thetaP"],ac["thetaD"]
    p = ac["efficacy"]
    Ca = ac["calciumVals"][ac["postNeuron"]]
    #from the Graupner and Brunel paper
    #rate = 1/tau 
    pStar = 0.5
    revP = (1.0-p)
    fp = -p*revP*(pStar-p)
    potentiation = gammaP*revP*(Ca>thetaP)
    depression = -gammaD*p*(thetaD<Ca)*(Ca<=thetaP)
    Noise = 0
    dp = rate*fp+potentiation+depression+Noise
    ac["efficacy"]+=TS*dp

def fastChannelUpdate(ac,TS,spikeOverwriteList):
    xs = ac["x"].flatten()
    xhs,xks = ac["vHalf"].flatten(),ac["vKi"].flatten()
    xts0,xts1 = ac["tau0"].flatten(),ac["tau1"].flatten()
    xths,xtks = ac["th"].flatten(),ac["tkInv"].flatten()
    V = ac["voltages"][ac["postNeuron"]]
    minOpenness = ac["minOpenness"].flatten()
    Ca = ac["calciumVals"][ac["postNeuron"]]

    #spikeOverwriteList = np.zeros(ac["nChannels"])

    synapseDelay = ac["inputDelay"]
    readIndex = ((ac["writeIndex"][ac["channelPreSpiker"]]-synapseDelay)%ac["maxDelayInt"])
    perceivedSpike = ac["spikeBuffer"][ac["channelPreSpiker"],readIndex]
    perceivedSpike = np.maximum(spikeOverwriteList,perceivedSpike)

    inputs = np.stack([V,V,Ca,perceivedSpike],-1,dtype=np.float32).flatten()
    res = solveGate(xs,inputs,TS,xhs,xks,xts0,xts1,xths,xtks,minOpenness)
    ac["x"] = res.reshape([-1,4])

def updateCurrents(ac):
    xs = ac["x"]
    exponents = ac["exponent"]

    gatingTotal = np.prod(xs ** exponents, axis=1)

    V = ac["voltages"][ac["postNeuron"]]
    Erev = ac["Erev"]
    g0 = ac["conductance"]
    p = ac["efficacy"]
    w = ac["conductanceGain"]
    CaCurrent = ac["CaCurrent"]

    g = g0 + w*p
    I = g*gatingTotal*(V-Erev) 
    ac["current"] = I
    ac["CaChannelCurrent"] = CaCurrent*I 
    ac["debugRecording"] = gatingTotal

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
        preSynapticSpiker = ac["channelPreSpiker"][i] #for channel this is = to postSynaptic neuron, and does not matter
        #spikes = max(spikeVals[preSynapticSpiker],spikeOverwriteList[i])
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
        #ac["netCurrent"][postSynapticNeuron] += I
        #ac["netCaCurrent"][postSynapticNeuron] += CaCurrent*I
        ac["current"][i] = I
        ac["CaChannelCurrent"][i] = CaCurrent*I #NOTE: Warning, we already have a variable called CaCurrent which tells you if a current is relevant to Ca change
        ac["debugRecording"][i] = gatingTotal #NOTE:27/7/25 change this in future, its just for debugging

        synapseDelay = ac["inputDelay"][i]
        readIndex = int((writeIndex[preSynapticSpiker]-synapseDelay)%maxDelay)
        perceivedSpike = ac["spikeBuffer"][preSynapticSpiker][readIndex]
        perceivedSpike = max(perceivedSpike,spikeOverwriteList[i])

        xs[0] = solveGate(xs[0],V,tempTS,xhs[0],xks[0],xts0[0],xts1[0],xths[0],xtks[0],minOpenness[0]) #gate1
        xs[1] = solveGate(xs[1],V,tempTS,xhs[1],xks[1],xts0[1],xts1[1],xths[1],xtks[1],minOpenness[1]) #gate2
        xs[2] = solveGate(xs[2],Ca,tempTS,xhs[2],xks[2],xts0[2],xts1[2],xths[2],xtks[2],minOpenness[2]) #CaGate
        xs[3] = solveGate(xs[3],perceivedSpike,tempTS,xhs[3],xks[3],xts0[3],xts1[3],xths[3],xtks[3],minOpenness[3]) #spikeGate

        dp = calcWeightChange(p,Ca,ac["rate"][i],ac["gammaP"][i],ac["gammaD"][i],ac["thetaP"][i],ac["thetaD"][i])
        ac["efficacy"][i]+=TS*dp


def solveDiffusiveConnectionsVectorized(ac):
    Vpost = ac["voltages"][ac["diffPostNeuron"]]
    Vpre = ac["voltages"][ac["diffPreNeuron"]]
    netDiffCurrent = ac["diffRate"]*(Vpost-Vpre)

    np.add.at(ac["netCurrent"],ac["diffPostNeuron"],netDiffCurrent)
    np.add.at(ac["netCurrent"],ac["diffPreNeuron"],-netDiffCurrent)

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
        #CaFlow = diffusionRate*(Capost-Capre) #technically this is electrodiffusion so I should probably implement that correctly 

        netCurrent[postSynapticNeuron] += I
        #netCaCurrent[postSynapticNeuron] += CaFlow*I #we are assuming Ca is flowing between compartments but this probably isnt the case due to sequestration

        netCurrent[preSynapticNeuron] -= I
        #netCaCurrent[preSynapticNeuron] -=s CaFlow*I

def solveNeurons(ac,biasOverWriteList,TS):
    #this function deals with determination of membrane voltage and calcium dynamics, 
    # i have moved spiking dynamics to the solveSpikers function 
    voltage = ac["voltages"]
    nSegments = ac["nSegments"]
    calciumVals = ac["calciumVals"]
    alpha = 1/(2*96485) #this is the conversion factor between current and Ca diffusion (1/zF). scale it up if you want Ca in different units 
    CaEquilibriumVal = ac["CaEquilibriumVal"] #1 I think 1e-6 is reasonble, otsuka is 0 
    CaRemovalRate = ac["CaRemovalRate"] #ms, this is the value otsuka uses, not sure where they get it from 
    for i in range(nSegments):
        voltage[i] -= TS*(ac["netCurrent"][i]+ac["biasCurrents"][i]+biasOverWriteList[i])
        calciumVals[i] += TS*(-alpha*ac["netCaCurrent"][i] + (-calciumVals[i]+ CaEquilibriumVal[i])*CaRemovalRate[i]) #This requires some explaination:
                        #We need to convert Ca current (in uA/cm^2) to Ca concentration (in mol/L) so I hardcode the conversion factor 
                        #Cellular Ca is tightly regulated to ~uM concentrations, so I've included a fixed point at 1e-6.
                        #I've set the timeconstant for Ca regulation to be 0.5ms, which might be a little fast
                        #With this implementation its possible to have 1. calcium efflux (which is fine), and 2. negative Ca concentrations (which is not fine)
                        #but negative Ca would require persistent Vm>140mV which is unphysiological anyway. 




def solveSpikers(ac,iteration):
    writeIndex = ac["writeIndex"]
    spikerPreNeuron = ac["spikerPreSegment"] #this is the neuron that gets checked 
    spikerNeuronScale = ac["spikerNeuronScale"] #0 or 1
    voltages = ac["voltages"]
    spikeBuffer = ac["spikeBuffer"]
    nSpikers = ac["nSpikers"]
    spikerProbability = ac["spikerProbability"]
    for i in range(nSpikers):
        randomOutput = bool(np.random.random()<spikerProbability[i])
        spikeOutput =  bool(spikerNeuronScale[i]*int(voltages[spikerPreNeuron[i]]>0))  # here we could actually implement transmitter release and short term plasticity dynamics 
        spikeOutput = float(randomOutput or spikeOutput)                      # for now its very simple spike threshold 
        spikeBuffer[i][writeIndex[i]] = spikeOutput         
        writeIndex[i] = (writeIndex[i]+1)%ac["maxDelayInt"]
        ac["spikerOutput"][i] = spikeOutput #This is just a recording variable, not used in any caclulations


def genSpikeStimArrays(stimInfoSpikes):
        spikeStimTimes = []
        spikeStimNeurons = []
        for key,val in stimInfoSpikes.items():
            for chan in val:
                spikeStimTimes.append(key)
                spikeStimNeurons.append(chan)
        spikeStimTimes,spikeStimNeurons=np.array(spikeStimTimes,dtype=np.int32), np.array(spikeStimNeurons,dtype=np.int32)
        spikeStimSortedInds = np.argsort(spikeStimTimes)
        spikeStimTimes,spikeStimNeurons = spikeStimTimes[spikeStimSortedInds],spikeStimNeurons[spikeStimSortedInds]
        return spikeStimTimes,spikeStimNeurons


def genBiasStimArrays(stimInfoBiasCurrent):
    biasStimTimes,biasStimSegments,biasStimValues = [],[],[]
    for key,val in stimInfoBiasCurrent.items():
        for seg,val in zip(val["segment"],val["value"]):
            biasStimTimes.append(key)
            biasStimSegments.append(seg)
            biasStimValues.append(val)

    biasStimTimes = np.array(biasStimTimes,dtype=np.int32)
    biasStimSegments = np.array(biasStimSegments,dtype=np.int32)
    biasStimValues = np.array(biasStimValues,dtype=np.float32)
    biasStimSortingInds = np.argsort(biasStimTimes)
    biasStimTimes = biasStimTimes[biasStimSortingInds]
    biasStimSegments = biasStimSegments[biasStimSortingInds]
    biasStimValues = biasStimValues[biasStimSortingInds]
    return biasStimTimes,biasStimSegments,biasStimValues

def solveSystemStandard(ac,TS,runTime,stimInfoSpikes={},stimInfoCurrents= {},vectorized=False,nSamples=None,useTQDM = False,
                        recordingKeys = ["voltages","calciumVals","netCurrent","current","spikeBuffer"]):
    nIterations = int(runTime/TS)
    ##Spike stim
    spikeStimTimes,spikeStimNeurons = genSpikeStimArrays(stimInfoSpikes)
    maxSpikeStimInd = len(spikeStimTimes)    
    spikeStimArrayIter = 0

    ##intracellular bias current stimulation
    biasStimTimes,biasStimSegments,biasStimValues = genBiasStimArrays(stimInfoCurrents)
    maxBiasStimInd = len(biasStimTimes)
    biasStimArrayIter = 0
    
    data = {key:[] for key in recordingKeys}
    tv = np.linspace(0,runTime,nIterations)
    nSegments = ac["nSegments"]
    nChannels = ac["nChannels"]

    if nSamples is None:
        samplingInterval = 1
    else:
        samplingInterval = max(1,int(nIterations/nSamples))
    
    tv = tqdm.tqdm(tv) if useTQDM else tv
    timeRecord = []
    for iteration,t in enumerate(tv):
        #calculate currents
        ac["netCurrent"] = np.zeros([nSegments])
        ac["netCaCurrent"] = np.zeros([nSegments])
        spikeOverwriteList = np.zeros(nChannels,dtype=np.float32)
        biasOverWriteList = np.zeros(nSegments,dtype=np.float32)

        while spikeStimArrayIter<maxSpikeStimInd and iteration == spikeStimTimes[spikeStimArrayIter]:
            syn = spikeStimNeurons[spikeStimArrayIter]
            spikeOverwriteList[syn] = 1
            spikeStimArrayIter +=1

        while biasStimArrayIter<maxBiasStimInd and iteration == biasStimTimes[biasStimArrayIter]:
            seg = biasStimSegments[biasStimArrayIter]
            val = biasStimValues[biasStimArrayIter]
            biasOverWriteList[seg] = val
            biasStimArrayIter +=1

        #update the gates
        #solveChannels(ac,spikeOverwriteList,TS)
        if vectorized:
            updateCurrents(ac)
            fastChannelUpdate(ac,TS,spikeOverwriteList)
            calcWeightChangeVectorized(ac,TS)
            solveDiffusiveConnectionsVectorized(ac)
        else:
            solveChannels(ac,spikeOverwriteList,TS)
            solveDiffusiveConnections(ac)

        #Accumulate current
        np.add.at(ac["netCurrent"],ac["postNeuron"],ac["current"])
        np.add.at(ac["netCaCurrent"],ac["postNeuron"],ac["CaChannelCurrent"])

        solveNeurons(ac,biasOverWriteList,TS)

        solveSpikers(ac,iteration)

        # for i in range(ac["nSpikers"]):
        #     if ac["spikerOutput"][i]>0:
        #         ac["spikeTrains"][i].append(iteration)

        if iteration%samplingInterval == 0:
            timeRecord.append(t)
            for key in data:
                data[key].append(deepcopy(ac[key]))

    for key in data:
        data[key] = np.array(data[key])
    data["time"] = np.array(timeRecord)
    return data

def addBasicChannelToSystem(allChannelData,neuronId,channelInfo):
    #This is the minimum base function for adding a channel-like mechanism to the system
    allChannelData["nChannels"]+=1  #not sure if i should do this at the start or end, probably doesnt matter
    allChannelData["channelName"].append(channelInfo["name"])
    allChannelData["postNeuron"].append(neuronId)
    allChannelData["channelPreSpiker"].append(0) #default is 0 if nothing else is provided later
    zeroKeys = ["efficacy","conductanceGain","rate","gammaP","gammaD","thetaP","thetaD",
                "inputDelay","current","CaChannelCurrent"]
    
    for key in zeroKeys:
        allChannelData[key].append(0)

    gates = channelInfo["gates"]
    gateSubKeys = gates[list(gates.keys())[0]].keys()
    #pull out the data into a list
    data = {gsk: [gv[gsk] for gv in gates.values()] for gsk in gateSubKeys}
    for key,newData in data.items():
        allChannelData[key].append(newData)

    allChannelData["Erev"].append(channelInfo["Erev"])
    allChannelData["conductance"].append(channelInfo["conductance"])
    allChannelData["CaCurrent"].append(channelInfo["CaCurrent"])
    # allChannelData["inputDelay"].append(0)
    # allChannelData["current"].append(0)

def addSynapseToSystem(preNeuron,postNeuron,synapseInfo,delayInt,allChannelData):
    addBasicChannelToSystem(allChannelData,postNeuron,synapseInfo)
    #allChannelData["postNeuron"][-1] = postNeuron 
    allChannelData["channelPreSpiker"][-1] = preNeuron
    allChannelData["inputDelay"][-1] = delayInt
    allChannelData["efficacy"][-1] = synapseInfo["efficacy"] if "efficacy" in synapseInfo.keys() else 0 #p, unitless, default efficacy is initialized to 0
    allChannelData["conductanceGain"][-1] = synapseInfo["conductanceGain"] if "conductanceGain" in synapseInfo.keys() else 0
    allChannelData["rate"][-1] = synapseInfo["rate"] if "rate" in synapseInfo.keys() else 0 #ms
    allChannelData["gammaP"][-1] = synapseInfo["gammaP"] if "gammaP" in synapseInfo.keys() else 0#p per ms
    allChannelData["gammaD"][-1] = synapseInfo["gammaD"] if "gammaD" in synapseInfo.keys() else 0#p per ms
    allChannelData["thetaP"][-1] = synapseInfo["thetaP"] if "thetaP" in synapseInfo.keys() else 0 #Ca mol/L
    allChannelData["thetaD"][-1] = synapseInfo["thetaD"] if "thetaD" in synapseInfo.keys() else 0 #Ca mol/L



def genGate(vHalf,vKi,tau0,tau1,th,tkInv,x0=0.0,exponent=1,minOpenness=0):
    return {"x":x0, "vHalf":vHalf,"vKi":vKi,"tau0":tau0,"tau1":tau1,"th":th,"tkInv":tkInv,"exponent":exponent,"minOpenness":minOpenness}

def genOpenGate():
    return genGate(-1000,1000,100,1000,0,0,x0=1)


def genSynapseGate(xt1,TS):
    xt0 = TS
    xt1 = xt1-xt0
    return genGate(0.5,100,xt0,xt1,0.5,-100)


def genDefaultPlasticSynapseData(**kwargs):
    plasticDict = {
        "efficacy": 0, #p, unitless
        "conductanceGain": 0, #
        "rate": 0.01, #ms
        "gammaP": 0.1, #p per ms
        "gammaD": 0.01, #p per ms
        "thetaP": 2e-5, #Ca mol/L
        "thetaD": 1e-5, #Ca mol/L
    }
    for k,v in kwargs.items():
        plasticDict[k] = v 
    return plasticDict

def genAMPA(TS,baseConductance,**plasticityKwargs):
    #for the synapses to work properly you need to know TS
    #This lets the conductance go instantly from 0 to 1
    AMPA = {"name":"AMPAR",
                "ion":"Cation",
            "Erev": 0,
            "conductance": baseConductance,
            "current":0,
            "CaCurrent": 0,
            "gates":{
            "vGate1":genOpenGate(),
            "vGate2":genOpenGate(),
            "CaGate":genOpenGate(),
            "spikeGate":genSynapseGate(7,TS)}} #decay is about 7ms for AMPA and GABA, though it can vary a lot, tbh this could be a random variable
    
    for key,val in genDefaultPlasticSynapseData(**plasticityKwargs).items():
        AMPA[key] = val

    return AMPA

def genNMDA_ion(TS,baseConductance,**plasticityKwargs):
    #NOTE: not sure if timescale is correct
    timeScale = 100
    NMDA = {"name":"NMDAR_ion",
                "ion":"Cation",
            "Erev": 0,
            "conductance": baseConductance,
            "current":0,
            "CaCurrent": 0,
            "gates":{
            "vGate1":genGate(-30,1/8,0.1,0.1,-1000,100),
            "vGate2":genOpenGate(),
            "CaGate":genOpenGate(),
            "spikeGate":genSynapseGate(timeScale,TS)}}
    for key,val in genDefaultPlasticSynapseData(**plasticityKwargs).items():
        NMDA[key] = val

    return NMDA

def genNMDA_Ca(TS,baseConductance,**plasticityKwargs):
    #NOTE: not sure if timescale is correct
    timeScale = 100
    NMDA = {"name":"NMDAR_Ca",
                "ion":"Cation",
            "Erev": 140, #ECa is very high, could be up to 140
            "conductance": baseConductance,
            "current":0,
            "CaCurrent": 1,
            "gates":{
            "vGate1":genGate(-30,1/8,0.1,0.1,-1000,100),
            "vGate2":genOpenGate(),
            "CaGate":genOpenGate(),
            "spikeGate":genSynapseGate(timeScale,TS)}}
    for key,val in genDefaultPlasticSynapseData(**plasticityKwargs).items():
        NMDA[key] = val

    return NMDA


def genGABA(TS,baseConductance,**plasticityKwargs):
    GABA = genAMPA(TS,baseConductance,**plasticityKwargs)
    GABA["name"] = "GABA"
    GABA["Erev"] = -80
    return GABA


def addSpikerToSystem(ac,spkrDict,spikerPreSegment):
    #all the stuff normally realting to neural spiking gets added here
    #spikes can be triggered either randomly and/or by spike thresholding in the presynaptic neuron
    #the simulation code checks for both conditions. To have only random spiking, choose an aribitrary presynaptic neuron (e.g. neuron 0),
    #and set the spikeNeuronScale = 0. To have only neuron spiking, set spikerProbability to 0.
    ac["spikerPreSegment"].append(spikerPreSegment)
    ac["spikerNeuronScale"].append(spkrDict["spikerNeuronScale"]) #0 for random, 1 for neuron based and random
    ac["spikerProbability"].append(spkrDict["spikerProbability"]) #set to 0 to remove random spiking
    ac["spikerOutput"].append(0)
    #ac["spikeTrains"].append([])


def addDiffusiveConnection(allChannelData,preNeuron,postNeuron,diffusionRate):
        allChannelData["diffPostNeuron"].append(postNeuron)
        allChannelData["diffPreNeuron"].append(preNeuron)
        allChannelData["diffRate"].append(diffusionRate)
        allChannelData["nDiffusiveConnections"]+=1

def getNeuronAndSpikerMaps(neuronDict,spikerDict):
    segmentIndex = 0
    spikerIndex = 0
    neuronPopToGlobalSegmentMap = {} #maps neurons onto the compartments containing their channels.
                                    #This is important for mapping synpatic currents onto compartments
    neuronPopToSpiker = {} #maps neurons onto their output spikers. Note this is NEURONs as a whole, not segments, and is used for connecting synapses
    spikerToPreSegment = {} #maps the spiker to the presynaptic segment that is used to produce the spike


    for neuronType,neurons in neuronDict.items():
        neuronPopToGlobalSegmentMap[neuronType] = {}
        neuronPopToSpiker[neuronType] = {}
        spikerToPreSegment[neuronType] = {}
         
        for neuron in neurons:
            neuronPopToGlobalSegmentMap[neuronType][neuron.id] = {}
            neuronPopToSpiker[neuronType][neuron.id] = spikerIndex
            spikerIndex +=1
            #the segment whos voltage is used to trigger a spike is the 0th segment. 
            spikerToPreSegment[neuronType][neuron.id] = segmentIndex 
            for segment,segmentInfo in neuron.segments.items():
                neuronPopToGlobalSegmentMap[neuronType][neuron.id][segment] = segmentIndex
                segmentIndex+=1

    for spikerType,spikerVal in spikerDict.items():
        neuronPopToSpiker[spikerType] = {}
        for spkr in spikerVal:
            neuronPopToSpiker[spikerType][spkr.id] = spikerIndex
            spikerIndex+=1
                

    return neuronPopToGlobalSegmentMap,neuronPopToSpiker,spikerToPreSegment


def genSimData(neuronDict,spikerDict,synapseDict,TS):
    #NOTE: 30/12/26 I am renaming spikerPreNeuron to spikerPreSegment because its more accurate and spikerPreNeuron was being used incorrectly
    #I have been referring to neuron segments as "neurons", because initialy this was a single compartment simulation
    #However I should probably start referring to them as "segments"
    simulationData = {key:[] for key in genOpenGate().keys()}
    listKeys = ["biasCurrents","CaEquilibriumVal","CaRemovalRate","neuronName", #neuron (segment) properties
                "postNeuron","channelPreSpiker","conductance","Erev","CaCurrent","inputDelay", #channel parameters
                "efficacy","conductanceGain","rate","gammaP","gammaD","thetaP","thetaD", #plasticity parameters
                "diffPostNeuron","diffPreNeuron","diffRate","current","CaChannelCurrent","channelName",#diffusion parameters
                "spikerPreSegment","spikerNeuronScale","spikerProbability","spikerOutput"] #spiker parameters 
    valueKeys = ["nChannels","nDiffusiveConnections","nSpikers","nSegments","nNeuralSpikers"] #nSegments used to be called nNeurons
    indexMaps = {}

    for key in listKeys:
        simulationData[key] = []
    for key in valueKeys:
        simulationData[key] = 0

    neuronPopToGlobalSegmentMap,neuronPopToSpiker,spikerToPreSegment = getNeuronAndSpikerMaps(neuronDict,spikerDict)
    indexMaps["neuronPopToGlobalSegmentMap"],indexMaps["neuronPopToSpikerMap"] = neuronPopToGlobalSegmentMap, neuronPopToSpiker
    indexMaps["spikerToPreSegmentMap"] = spikerToPreSegment
    neuronToChannelMap = {}
    for neuronType,neurons in neuronDict.items():
        neuronToChannelMap.setdefault(neuronType,{})
        for neuron in neurons:
            AISid = spikerToPreSegment[neuron.name][neuron.id] #the segment whos voltage is used to trigger a spike 
            addSpikerToSystem(simulationData,neuron.spiker,AISid)
            simulationData["nNeuralSpikers"] +=1
            neuronToChannelMap[neuronType].setdefault(AISid,{})

            for segment,segmentInfo in neuron.segments.items():
                neuronToChannelMap[neuronType][AISid].setdefault(segment,{})
                for channelName,channelData in segmentInfo["channels"].items():
                    neuronToChannelMap[neuronType][AISid][segment][channelName] = simulationData['nChannels'] #nChannels is updated whenever a channel is added
                    globalId = neuronPopToGlobalSegmentMap[neuron.name][neuron.id][segment]
                    addBasicChannelToSystem(simulationData,globalId,channelData)

                simulationData["biasCurrents"].append(segmentInfo["bias"])
                simulationData["CaEquilibriumVal"].append(segmentInfo["CaEquilibriumVal"])
                simulationData["CaRemovalRate"].append(segmentInfo["CaRemovalRate"])
                simulationData["neuronName"].append(f"{neuronType}_{neuron.id}_{segment}")

    indexMaps['neuronToChannelMap'] = neuronToChannelMap

    for spikerName,spikerVal in spikerDict.items():
        for spkr in spikerVal:
            addSpikerToSystem(simulationData,spkr.spiker,0)

    nSegments = len(simulationData["biasCurrents"])
    nSpikers = len(simulationData["spikerPreSegment"])
    simulationData["nSegments"] = nSegments
    simulationData["nSpikers"] = nSpikers
    #############################################################################################
    #efficacy, conductanceGain, rate, gammaP, gammaD, thetaP, thetaD,
    
    simulationData["writeIndex"] = np.zeros(nSpikers,dtype=np.int32)
    maxDelayInt = 1 #default to 1
    connectionToChannelMap = {} #it might be more elegant to have this somewhere outside this function, but it'll be fine here for now
                                #also we might want to do different indexing in future, e.g. connectionToChannelMap[synapseGroup][preNeuron][postNeuron][synType]
    for synapseGroupName,synapseGroupData in synapseDict.items():
        connectionToChannelMap[synapseGroupName] = {}
        for connInd,connection in enumerate(synapseGroupData):
            connectionToChannelMap[synapseGroupName][connInd] = simulationData["nChannels"]
            preLocalIndex,postLocalIndex = connection["pre"],connection["post"]
            preName,postName = connection["preName"],connection["postName"]
            segmentIndex = connection["segmentIndex"]
            preSynapticSpikerIndex = neuronPopToSpiker[preName][preLocalIndex] #pick the spiker of the pre-synaptic neuron
            postSynapticSegmentIndex = neuronPopToGlobalSegmentMap[postName][postLocalIndex][segmentIndex] #have it connect to post synaptic segment
            synapseData = connection["synapse"]
            delay = connection["delay"] #in ms
            delayInt = int(np.ceil(delay/TS))
            maxDelayInt = max(delayInt,maxDelayInt)
            assert 0<=preSynapticSpikerIndex<nSpikers and 0<=postSynapticSegmentIndex<nSegments, f"edge out of range {preSynapticSpikerIndex},{postSynapticSegmentIndex}"
            addSynapseToSystem(preSynapticSpikerIndex,postSynapticSegmentIndex,synapseData,delayInt,simulationData)
    
    indexMaps["connectionToChannelMap"] = connectionToChannelMap


    #Diffusive connections
    for neuronType,neurons in neuronDict.items():
        for neuron in neurons:
            for edge in neuron.segmentNetwork:
                preSegLocalInd,postSegLocalInd,diffusionRate = edge
                preSegmentIndex = neuronPopToGlobalSegmentMap[neuronType][neuron.id][preSegLocalInd]
                postSegmentIndex = neuronPopToGlobalSegmentMap[neuronType][neuron.id][postSegLocalInd]
                assert preSegmentIndex<nSegments and postSegmentIndex<nSegments, f"edge out of range {preSegmentIndex},{postSegmentIndex}"
                addDiffusiveConnection(simulationData,preSegmentIndex,postSegmentIndex,diffusionRate)

    nextPower2 = lambda x: int(2**(np.ceil(np.log2(x))))
    maxDelayInt = nextPower2(maxDelayInt)
    simulationData["maxDelayInt"] = maxDelayInt
    simulationData["voltages"] = -75*np.ones(nSegments,dtype = np.float32) + 1*np.linspace(-10,20,nSegments, dtype=np.float32)
    simulationData["calciumVals"] = np.zeros(nSegments,dtype = np.float32)
    simulationData["spikeVals"] = np.zeros(nSpikers,dtype = np.float32)
    simulationData["netCurrent"] = np.zeros(nSegments,dtype = np.float32)
    simulationData["netCaCurrent"] = np.zeros(nSegments,dtype = np.float32)
    simulationData["spikeBuffer"] = np.zeros((nSpikers,maxDelayInt),dtype=bool) #we could use float if we wanted some kind of scaling, but for perf we use bool

    intKeys = ["channelPreSpiker","postNeuron", "nChannels","diffPostNeuron",
               "diffPreNeuron","nDiffusiveConnections","writeIndex",
               "maxDelayInt","inputDelay","nSegments","nSpikers","spikerPreSegment",
               ]
    
    nonArrayKeys = ["nSegments","nNeuralSpikers","maxDelayInt","nChannels","nDiffusiveConnections","channelName","nSpikers","neuronName"]
    for key in simulationData.keys():
        val = simulationData[key]
        if key in intKeys: arrDtype = np.int32
        else: arrDtype = np.float32

        if key in nonArrayKeys:
            pass
        elif type(val) in [list,tuple]:
            simulationData[key] = np.ascontiguousarray(val,dtype = arrDtype)
        elif type(val)  == np.ndarray:
            simulationData[key] = np.ascontiguousarray(val,dtype = val.dtype)
        else:
            print(f"Warning {key} is not a list or tuple") 
    
    simulationData["exponent"] = np.array(simulationData["exponent"],dtype=np.int8)
    simulationData["indexMaps"] = indexMaps
    
    return simulationData


class Neuron():
    def __init__(self,index,name,neuronType,AISchannels,AISbias,denderiteInfo={},dendriteMap=[],CaEquilibriumVal=0,CaRemovalRate=2):
        neuronTypeToSynapseList = {"excitatory":["AMPA","NMDA_ion","NMDA_Ca"],#,
                                   "inhibitory":["GABA"]}
        self.id = index
        self.name=name
        self.outgoingSynapses = neuronTypeToSynapseList[neuronType]
        self.segments = {0:{"channels":AISchannels,"bias":AISbias,"CaEquilibriumVal":CaEquilibriumVal,
                            "CaRemovalRate":CaRemovalRate}}
        for i,dInfo in enumerate(denderiteInfo):
            self.segments[i+1] = deepcopy(dInfo)
        self.segmentNetwork = deepcopy(dendriteMap)
        self.spiker = {"spikerNeuronScale":1,"spikerProbability":0}
        #CaEquilibriumVal #equilib value of calcium in mM/L (otsuka neurons assume its 0, in reality its around 1e-6)
        #CaRemovalRate #timescale on which calcium is removed (typically 2ms)
        
        
class Spiker():
    def __init__(self,index,name,neuronType,spikeRate,TS):
        neuronTypeToSynapseList = {"excitatory":["AMPA","NMDA_ion","NMDA_Ca"],#,"NMDA"
                                   "inhibitory":["GABA"]}
        
        self.id = index
        self.name = name
        self.outgoingSynapses = neuronTypeToSynapseList[neuronType]
        self.spiker = {"spikerNeuronScale":0,"spikerProbability":spikeRate*TS/1000.0}


def createSynapses(prePopulation,postPopulation,nSynapses,baseConductance,delay,TS,synapses=None, edgeList=None):
    if synapses == None:
        #default synapses
        synapses = {"AMPA":genAMPA(TS,baseConductance,conductanceGain=0),
                    "NMDA_ion":genNMDA_ion(TS,baseConductance,conductanceGain=0),
                    "NMDA_Ca":genNMDA_Ca(TS,baseConductance,conductanceGain=0),
                    "GABA":genGABA(TS,baseConductance,conductanceGain=0)}
    
    if edgeList is None:
        nPre,nPost = len(prePopulation),len(postPopulation)
        preVals = np.random.randint(0,nPre,nSynapses)
        postVals = np.random.randint(0,nPost,nSynapses)
        segmentIndices = [np.random.randint(len(postPopulation[e1].segments)) for e1 in postVals]
        edgeList = [(e0,e1,s) for e0,e1,s in zip(preVals,postVals,segmentIndices)]

    synapseTypes = prePopulation[0].outgoingSynapses
    preName = prePopulation[0].name
    postName = postPopulation[0].name
    synapseData = []

    for edge in edgeList:
        e0,e1,segmentIndex = edge[0],edge[1],edge[2]
        for synapse in synapseTypes:
            synapseInfo = synapses[synapse]
            synDat = {"pre":e0,"post":e1,"synapse":deepcopy(synapseInfo),"delay":delay,"preName":preName,"postName":postName,"segmentIndex":segmentIndex}
            synapseData.append(synDat)

    return synapseData


def stimSynapse(stimInfoSpikes,synapsesToStim,startTime,stopTime,TS):
    #another option is to have a stim value as well. We assume here stim value is 1, but
    #if we could set it to 0 we could switch off ion channels
    st0,st1 = int(startTime/TS),int(stopTime/TS)
    times = [i for i in range(st0,st1)]
    for t in times:
        for syn in synapsesToStim:
            if t not in stimInfoSpikes:
                stimInfoSpikes[t] = set([syn])
            stimInfoSpikes[t].add(syn)
                

def stimBiasCurrent(stimInfoBiasCurrent,segmentsToStim,currentValue, startTime,stopTime,TS):
    #another option is to have a stim value as well. We assume here stim value is 1, but
    #if we could set it to 0 we could switch off ion channels
    st0,st1 = int(startTime/TS),int(stopTime/TS)
    times = [i for i in range(st0,st1)]
    for t in times:
        for seg in segmentsToStim:
            if t not in stimInfoBiasCurrent:
                stimInfoBiasCurrent[t] = {"segment":[],"value":[]}
            stimInfoBiasCurrent[t]["segment"].append(seg)
            stimInfoBiasCurrent[t]["value"].append(currentValue)