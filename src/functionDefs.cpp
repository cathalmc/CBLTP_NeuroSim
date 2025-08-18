#include "functionDefs.hpp"
#include <execution>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <omp.h>

int ompMaxThreads() { return omp_get_max_threads(); }

long long paritySum(long long N){
    long long s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(long long i=0;i<N;++i) s += (i & 1);
    return s;
}

struct Xoshiro32 {
    uint32_t s[4];
    static inline uint32_t rotl(uint32_t x, int k){ return (x<<k)|(x>>(32-k)); }
    explicit Xoshiro32(uint64_t seed){
        auto splitmix = [](uint64_t& x){
            x += 0x9E3779B97F4A7C15ull;
            uint64_t z = x;
            z = (z ^ (z>>30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z>>27)) * 0x94D049BB133111EBull;
            return z ^ (z>>31);
        };
        uint64_t x = seed;
        for (int i=0;i<4;++i) s[i] = static_cast<uint32_t>(splitmix(x));
        if (!(s[0]|s[1]|s[2]|s[3])) s[0] = 1;
    }
    inline uint32_t nextU32(){
        uint32_t r = rotl(s[1]*5u,7)*9u;
        uint32_t t = s[1] << 9;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
        s[2]^=t; s[3]=rotl(s[3],11);
        return r;
    }
    inline float nextFloat(){ return (nextU32() >> 8) * (1.0f/16777216.0f); } // 24-bit mantissa
};

PYBIND11_MODULE(cathalTest, m) {
    m.def("testFunc",&testFunc,"run my lil test function",
    "testIntVal"_a,"testNParr"_a);
    m.def("solveNeuralSystem",&solveNeuralSystem,"test accepting dictionaries",
    "myDict"_a,"recordingInfo"_a, "timeStep"_a,"nIterations"_a,"samplingInterval"_a);
    m.def("ompMaxThreads", &ompMaxThreads);
    m.def("paritySum", &paritySum, py::call_guard<py::gil_scoped_release>()); // release GIL
}

inline float sigmoid(float x, float xh, float xk,float minOp){
    return minOp + (1.0f/(1.0f + expf((xh-x)*xk))); //normally you divide by K, but I precalculate the inverse to save on computation. 
    //However debugging this has cost me so much time that it probably wasn't worth it.
}

inline float solveGate(float x, float s, float TS, float xh, float xk, float xts0, 
        float xts1, float xth, float xtk,float minOpenness){
    float xInf = sigmoid(s,xh,xk,minOpenness);
    float tau = xts0 + xts1*sigmoid(s,xth,xtk,0.0f);
    //return x + TS*(xInf-x)/tau;
    return (tau*x +TS*xInf)/(tau+TS);
}

static inline float powiFast(float x, int e){
    // non-negative e (store exponents as int8_t/uint8_t)
    // for gating exponents, the max power is 4, so maybe we could limit this
    switch(e){
        case 0: return 1.0f;
        case 1: return x;
        case 2: return x*x;
        case 3: return x*x*x;
        case 4: { float x2=x*x; return x2*x2; }
        default:{ float r=1.0f; while(e){ if(e&1) r*=x; x*=x; e>>=1; } return r; }
    }
}
static inline float sigmoidInl(float s, float xh, float xk, float minOp){
    float z = expf((xh - s) * xk);
    return minOp + 1.0f/(1.0f + z);
}
static inline float gateStep(float xOld, float s, float TS,
                             float xh, float xk, float ts0, float ts1,
                             float th, float tkInv, float minOp){
    float xInf = sigmoidInl(s,   xh,  xk,  minOp);
    float tau  = ts0 + ts1 * sigmoidInl(s, th, tkInv, 0.0f);
    float inv  = 1.0f / (tau + TS);
    return fmaf(TS, xInf, tau * xOld) * inv;
}

// inline float calcWeightChange(float p, float Ca, float rate, float gammaP, float gammaD, float thetaP, float thetaD){
//     float pStar = 0.5;
//     float revP = (1-p);
//     float fp = -p*revP*(pStar-p);
//     float potentiation = gammaP*revP*(Ca>thetaP);
//     float depression = -gammaD*p*(thetaD<Ca && Ca<=thetaP);
//     float Noise = 0;
//     return rate*fp+potentiation+depression+Noise;
// }
inline float calcWeightChange(float p, float Ca, float rate,
                              float gammaP, float gammaD,
                              float thetaP, float thetaD){
    float pStar=0.5f, revP=(1.f-p);
    float fp = -p*revP*(pStar-p);
    float mP = Ca>thetaP;                // 0 or 1
    float mD = (Ca>thetaD) & (Ca<=thetaP); // 0 or 1 without branch
    return rate*fp + gammaP*revP*mP - gammaD*p*mD;
}


py::object solveNeuralSystem(py::dict systemInfo,py::dict recordingInfo,
                            float timeStep,int nIterations,int samplingInterval) {
    //systemInfo is a dict of numpy arrays, we need to extract these arrays and process them in place
    std::random_device rd;  // seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> uniformFloatDist(0.0f, 1.0f);

    constexpr float alpha = 1.0/(2.0*96485); //converts current to calcium diffusion, 
    // constexpr float CaEquilibriumVal = 0.0f; // should be 1e-6, but keeping it 0 for consistency with otsuka
    // constexpr float CaRemovalRate = 2.0f; //from otsuka
    //Single values 
    int nNeurons = systemInfo["nNeurons"].cast<int>();
    int nSpikers = systemInfo["nSpikers"].cast<int>();
    int nChannels = systemInfo["nChannels"].cast<int>();
    int nDiffusiveConnections = systemInfo["nDiffusiveConnections"].cast<int>();
    int maxDelayInt = systemInfo["maxDelayInt"].cast<int>();

    //Neuron data
    auto biasCurrentsBuffer = systemInfo["biasCurrents"].cast<py::array_t<float>>().request();
    float *biasCurrents = (float*)(biasCurrentsBuffer.ptr);
    auto CaEquilibriumValBuffer = systemInfo["CaEquilibriumVal"].cast<py::array_t<float>>().request();
    float *CaEquilibriumVal = (float*)(CaEquilibriumValBuffer.ptr);
    auto CaRemovalRateBuffer = systemInfo["CaRemovalRate"].cast<py::array_t<float>>().request();
    float *CaRemovalRate = (float*)(CaRemovalRateBuffer.ptr);

    //Neuron data
    auto spikerPreNeuronBuffer = systemInfo["spikerPreNeuron"].cast<py::array_t<int>>().request();
    int *spikerPreNeuron = (int*)(spikerPreNeuronBuffer.ptr);
    auto spikerNeuronScaleBuffer = systemInfo["spikerNeuronScale"].cast<py::array_t<float>>().request();
    float *spikerNeuronScale = (float*)(spikerNeuronScaleBuffer.ptr);
    auto spikerProbabilityBuffer = systemInfo["spikerProbability"].cast<py::array_t<float>>().request();
    float *spikerProbability = (float*)(spikerProbabilityBuffer.ptr);

    //bool arrays
    auto spikeBufferBuffer = systemInfo["spikeBuffer"].cast<py::array_t<bool, py::array::c_style>>().request();
    bool* spikeBuffer = (bool *) spikeBufferBuffer.ptr;
    int psuedoMaxDelayInt = spikeBufferBuffer.shape[1];
    size_t spikeBufferNeuronStride = spikeBufferBuffer.strides[0]/sizeof(bool);
    size_t spikeBufferTimeStride = spikeBufferBuffer.strides[1]/sizeof(bool);
    

    //assert(maxDelayInt==psuedoMaxDelayInt); //is this best practice in c++?
    if (maxDelayInt != psuedoMaxDelayInt)
    throw std::runtime_error("maxDelayInt mismatch detected!");
    
    //integer arrays
    //intKeys = ["preNeuron","postNeuron","diffPostNeuron","diffPreNeuron","writeIndex", "inputDelay"]
    auto channelPreSpikerBuffer = systemInfo["channelPreSpiker"].cast<py::array_t<int>>().request();
    int *channelPreSpiker = static_cast<int*>(channelPreSpikerBuffer.ptr);
    auto postNeuronBuffer = systemInfo["postNeuron"].cast<py::array_t<int>>().request();
    int *postNeuron = static_cast<int*>(postNeuronBuffer.ptr);
    auto diffPreNeuronBuffer = systemInfo["diffPreNeuron"].cast<py::array_t<int>>().request();
    int *diffPreNeuron = static_cast<int*>(diffPreNeuronBuffer.ptr);
    auto diffPostNeuronBuffer = systemInfo["diffPostNeuron"].cast<py::array_t<int>>().request();
    int *diffPostNeuron = static_cast<int*>(diffPostNeuronBuffer.ptr);
    auto writeIndexBuffer = systemInfo["writeIndex"].cast<py::array_t<int>>().request();
    int *writeIndex = static_cast<int*>(writeIndexBuffer.ptr);
    auto inputDelayBuffer = systemInfo["inputDelay"].cast<py::array_t<int>>().request();
    int *inputDelay = static_cast<int*>(inputDelayBuffer.ptr);


    //Ion channel data
    //keys: [''voltages', 'calciumVals', 'spikeVals', "CaCurrent", 
    auto voltageBuffer = systemInfo["voltages"].cast<py::array_t<float>>().request();
    float * voltages = (float *) voltageBuffer.ptr;
    if (voltageBuffer.size < nNeurons) throw std::runtime_error("voltages array too small");
    auto diffRateBuffer = systemInfo["diffRate"].cast<py::array_t<float>>().request();
    float * diffRate = (float *) diffRateBuffer.ptr;
    auto calciumValsBuffer = systemInfo["calciumVals"].cast<py::array_t<float>>().request();
    float * calciumVals = (float *) calciumValsBuffer.ptr;
    auto spikeValsBuffer = systemInfo["spikeVals"].cast<py::array_t<float>>().request();
    float * spikeVals = (float *) spikeValsBuffer.ptr;
    auto isThisCaCurrentBuffer = systemInfo["CaCurrent"].cast<py::array_t<float>>().request();
    float * isThisCaCurrent = (float *) isThisCaCurrentBuffer.ptr;
    auto netCurrentBuffer = systemInfo["netCurrent"].cast<py::array_t<float>>().request();
    float * netCurrent = (float *) netCurrentBuffer.ptr;
    auto netCaCurrentBuffer = systemInfo["netCaCurrent"].cast<py::array_t<float>>().request();
    float * netCaCurrent = (float *) netCaCurrentBuffer.ptr;
    auto channelCurrentBuffer = systemInfo["current"].cast<py::array_t<float>>().request();
    float * channelCurrent = (float *) channelCurrentBuffer.ptr;
    auto CaChannelCurrentBuffer = systemInfo["CaChannelCurrent"].cast<py::array_t<float>>().request();
    float * CaChannelCurrent = (float *) CaChannelCurrentBuffer.ptr;
    auto debugRecordingBuffer = systemInfo["debugRecording"].cast<py::array_t<float>>().request();
    float * debugRecording = (float *) debugRecordingBuffer.ptr;
    
    //gating data 
    //x', 'vHalf', 'vKi', 'tau0', 'tau1', 'th', 'tkInv', 'exponent',"minOpenness" , 
    auto xBuffer = systemInfo["x"].cast<py::array_t<float>>().request(); //.ptr;
    //TODO: consider changing this data structure to be multiple separate arrays
    size_t nGatingVariables = xBuffer.shape[1];
    size_t chanStride = xBuffer.strides[0]/sizeof(float);
    size_t gateStride = xBuffer.strides[1]/sizeof(float);
    float *x = (float*) xBuffer.ptr;

    auto vHalfBuffer = systemInfo["vHalf"].cast<py::array_t<float>>().request();
    float *vHalf = (float *) vHalfBuffer.ptr;
    auto vKiBuffer = systemInfo["vKi"].cast<py::array_t<float>>().request();
    float *vKi = (float *) vKiBuffer.ptr;
    auto tau0Buffer = systemInfo["tau0"].cast<py::array_t<float>>().request();
    float *tau0 = (float *) tau0Buffer.ptr;
    auto tau1Buffer = systemInfo["tau1"].cast<py::array_t<float>>().request();
    float *tau1 = (float *) tau1Buffer.ptr;
    auto thBuffer = systemInfo["th"].cast<py::array_t<float>>().request();
    float *th = (float *) thBuffer.ptr;
    auto tkInvBuffer = systemInfo["tkInv"].cast<py::array_t<float>>().request();
    float *tkInv = (float *) tkInvBuffer.ptr;
    auto exponentBuffer = systemInfo["exponent"].cast<py::array_t<int8_t>>().request();
    int8_t *exponent = (int8_t *) exponentBuffer.ptr;
    auto minOpennessBuffer = systemInfo["minOpenness"].cast<py::array_t<float>>().request();
    float *minOpenness = (float *) minOpennessBuffer.ptr;


    //Synapse data
    // "rate","gammaP","gammaD", "thetaP","thetaD",'conductance',"efficacy","conductanceGain" 'Erev',  ]
    auto rateBuffer = systemInfo["rate"].cast<py::array_t<float>>().request();
    float *rate = (float *) rateBuffer.ptr;
    auto gammaPBuffer = systemInfo["gammaP"].cast<py::array_t<float>>().request();
    float *gammaP = (float *) gammaPBuffer.ptr;
    auto gammaDBuffer = systemInfo["gammaD"].cast<py::array_t<float>>().request();
    float *gammaD = (float *) gammaDBuffer.ptr;
    auto thetaPBuffer = systemInfo["thetaP"].cast<py::array_t<float>>().request();
    float *thetaP = (float *) thetaPBuffer.ptr;
    auto thetaDBuffer = systemInfo["thetaD"].cast<py::array_t<float>>().request();
    float *thetaD = (float *) thetaDBuffer.ptr;
    auto conductanceBuffer = systemInfo["conductance"].cast<py::array_t<float>>().request();
    float *conductance = (float *) conductanceBuffer.ptr;
    auto efficacyBuffer = systemInfo["efficacy"].cast<py::array_t<float>>().request();
    float *efficacy = (float *) efficacyBuffer.ptr;
    auto conductanceGainBuffer = systemInfo["conductanceGain"].cast<py::array_t<float>>().request();
    float *conductanceGain = (float *) conductanceGainBuffer.ptr;
    auto ErevBuffer = systemInfo["Erev"].cast<py::array_t<float>>().request();
    float *Erevs = (float *) ErevBuffer.ptr;

    //Stimulation data 
    // Spike stimulation
    std::vector<float> spikeInputOverwriteVector(nChannels, 0.0f);
    auto spikeStimTimesBuffer = systemInfo["spikeStimTimes"].cast<py::array_t<int>>().request();
    int *spikeStimTimes = (int*)(spikeStimTimesBuffer.ptr);
    auto spikeStimNeuronsBuffer = systemInfo["spikeStimNeurons"].cast<py::array_t<int>>().request();
    int *spikeStimNeurons = (int*)(spikeStimNeuronsBuffer.ptr);
    size_t maxSpikeStimInd = spikeStimTimesBuffer.shape[0];
    size_t spikeStimArrayIter = 0; //used to iterate through the spike stimulation arrays
    
    //Bias current stimulation
    std::vector<float> biasCurrentOverwriteVector(nNeurons, 0.0f);
    //have biasStimTimes, biasStimSegments biasStimValues
    auto biasStimTimesBuffer = systemInfo["biasStimTimes"].cast<py::array_t<int>>().request();
    int *biasStimTimes = (int*)(biasStimTimesBuffer.ptr);
    auto biasStimSegmentsBuffer = systemInfo["biasStimSegments"].cast<py::array_t<int>>().request();
    int *biasStimSegments = (int*)(biasStimSegmentsBuffer.ptr);
    auto biasStimValuesBuffer = systemInfo["biasStimValues"].cast<py::array_t<float>>().request();
    float *biasStimValues = (float*)(biasStimValuesBuffer.ptr);
    size_t maxBiasStimInd = biasStimTimesBuffer.shape[0];
    size_t biasStimArrayIter = 0; //used to iterate through the bias stimulation arrays

    auto performanceStatsBuffer = systemInfo["performanceStats"].cast<py::array_t<float>>().request();
    float *performanceStats = (float *)(performanceStatsBuffer.ptr);

    //float gatingTotal;
    //assign temporary variables for readability
    //float I,V,Ca,xs,xhs,xks,xts0,xts1,xths,xtks,Erev,g,gateVar,perceivedSpike;
    //size_t preSynapticNeuron,postSynapticNeuron,readIndex;

    size_t sampleInd = 0;
    
    std::vector<int> indices(nChannels);
    std::iota(indices.begin(), indices.end(), 0); // Fill it with consecutive numbers
    //auto executionPolicy = std::execution::seq;
    // std::for_each(executionPolicy, indices.begin(), indices.end(), [&](int connInd)  {
    // });

    //Optimizations
    omp_set_num_threads(28);
    
    const int mask = maxDelayInt - 1;
    // A changing base seed for RNG reproducibility across time steps if desired
    const uint64_t baseSeed = 0xA3C59AC3DUL;

    //Optimizing the accumulate section 
    std::vector<int> neuronOffsets(nNeurons+1, 0);
    for (int c=0;c<nChannels;++c) ++neuronOffsets[postNeuron[c]];
    // prefix sum -> offsets
    int run=0; for (int n=0;n<=nNeurons;++n){ int t=neuronOffsets[n]; neuronOffsets[n]=run; run+=t; }
    std::vector<int> chanIndexByNeuron(nChannels);
    // fill buckets
    std::vector<int> cursor = neuronOffsets; // copy
    for (int c=0;c<nChannels;++c) chanIndexByNeuron[cursor[postNeuron[c]]++] = c;



    auto executionPolicy = std::execution::par_unseq; //par_unseq
    double initializationT,channelT,accumulateT,diffusiveT, spikerT,neuronT, recordingT;
    initializationT = 0.0, channelT = 0.0, accumulateT = 0.0, diffusiveT = 0.0, spikerT = 0.0, neuronT = 0.0, recordingT = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t timeInd=0;timeInd<nIterations;timeInd++){
        auto innerLoopStart = std::chrono::steady_clock::now();
        for (size_t idx=0;idx<nNeurons;idx++){
            netCurrent[idx] = 0.0;
            netCaCurrent[idx] = 0.0;
        }
        //Spike overwrite code needs to be implemented here 
        std::fill(spikeInputOverwriteVector.begin(), spikeInputOverwriteVector.end(), 0.0f);
        // Spike overwrite:
        // Create spikeStimTimes,spikeStimNeurons vectors, and use a while loop with those
        while (spikeStimArrayIter<maxSpikeStimInd && timeInd == spikeStimTimes[spikeStimArrayIter])
        {
            int syn = spikeStimNeurons[spikeStimArrayIter];
            spikeInputOverwriteVector[syn] = 1.0f;
            spikeStimArrayIter++;
        }

        //Bias current overwrite 
        std::fill(biasCurrentOverwriteVector.begin(), biasCurrentOverwriteVector.end(), 0.0f);
        while (biasStimArrayIter<maxBiasStimInd && timeInd == biasStimTimes[biasStimArrayIter])
        {
            int neuron = biasStimSegments[biasStimArrayIter];
            float biasVal = biasStimValues[biasStimArrayIter];
            biasCurrentOverwriteVector[neuron] = biasVal;
            biasStimArrayIter++;
        }
        auto initializationEnd = std::chrono::steady_clock::now();
        initializationT += std::chrono::duration<double>(initializationEnd - innerLoopStart).count();
        
        //Solve Channels 
        // std::iota(indices.begin(), indices.end(), 0); // Fill it with consecutive numbers
        // std::for_each(executionPolicy, indices.begin(), indices.end(), [&](int connInd)  {
        // #pragma omp parallel for schedule(dynamic, 256)
        // for (long long int connInd = 0; connInd<nChannels; connInd++){
        //     //temporary variables
        //     long long int idx = connInd;
        //     int postSynapticNeuron = postNeuron[idx];
        //     int preSynapticNeuron = channelPreSpiker[idx];
        //     float V = voltages[postSynapticNeuron];
        //     float Ca = calciumVals[postSynapticNeuron];
        //     float g = conductance[idx] + conductanceGain[idx]*efficacy[idx];

        //     //multiply gating variables to get net gate opening
        //     float gatingTotal = 1.0f;
        //     for (size_t idy=0;idy<nGatingVariables;idy++){
        //         float expnt = exponent[idx*chanStride + idy];
        //         gatingTotal *= pow(x[idx*chanStride + idy],expnt);
        //     }

        //     float I = g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
        //     channelCurrent[idx] = I; //we accumulate later
        //     CaChannelCurrent[idx] = isThisCaCurrent[idx]*I;
        //     debugRecording[idx] = gatingTotal;

        //     const int mask = maxDelayInt-1;
        //     int readIndex = (writeIndex[preSynapticNeuron] - inputDelay[idx]) & mask;
        //     //readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
        //     float perceivedSpike = (float)spikeBuffer[preSynapticNeuron*spikeBufferNeuronStride+readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
        //     perceivedSpike = std::max(perceivedSpike,spikeInputOverwriteVector[idx]); //stimulateion 

        //     float dp = calcWeightChange(efficacy[idx],Ca,rate[idx],gammaP[idx],gammaD[idx],thetaP[idx],thetaD[idx]);
        //     efficacy[idx] +=timeStep*dp;
        //     float gateVar;
        //     for (size_t idy=0;idy<nGatingVariables;idy++) {
        //         if (idy<2) gateVar = V;
        //         else if (idy==2) gateVar = Ca; //Ca
        //         else gateVar = perceivedSpike; //Spikes

        //         float xp = x[idx*chanStride + idy];
        //         float vHalfp = vHalf[idx*chanStride + idy];
        //         float vKip = vKi[idx*chanStride + idy];
        //         float tau0p = tau0[idx*chanStride + idy];
        //         float tau1p = tau1[idx*chanStride + idy];
        //         float thp = th[idx*chanStride + idy];
        //         float tkInvp = tkInv[idx*chanStride + idy];
        //         float minOp = minOpenness[idx*chanStride + idy];

        //         x[idx*chanStride + idy] = solveGate(xp,gateVar,timeStep,vHalfp,vKip,tau0p,tau1p,thp,tkInvp,minOp);


        //     }

        // }
        //);


        const int mask = maxDelayInt - 1;

        // alias “rows” once per channel (hoist) to cut address calc
        #pragma omp parallel for schedule(dynamic, 256)   // or static; dynamic with bigger chunk if you prefer
        for (long long connInd = 0; connInd < nChannels; ++connInd){
            const int idx  = (int)connInd;
            const int post = postNeuron[idx];
            const int pre  = channelPreSpiker[idx];

            const float V  = voltages[post];
            const float Ca = calciumVals[post];

            // ---- hoist base pointers for this channel (SoA rows) ----
            const int base = idx * chanStride;
            // state vector (x) and parameters per gate (fixed 4 gates: V0,V1,Ca,Spike)
            float      * __restrict xRow    = &x[base];             // [0..3]
            const int8_t* __restrict eRow   = &exponent[base];    // int8 exponents
            const float * __restrict vHalfR = &vHalf[base];
            const float * __restrict vKiR   = &vKi[base];
            const float * __restrict tau0R  = &tau0[base];
            const float * __restrict tau1R  = &tau1[base];
            const float * __restrict thR    = &th[base];
            const float * __restrict tkInvR = &tkInv[base];
            const float * __restrict minOpR = &minOpenness[base];

            // ---- perceived spike (bool 2D flattened; respect both strides) ----
            int wread = (writeIndex[pre] - inputDelay[idx]) & mask;
            bool sbit = spikeBuffer[(size_t)pre * spikeBufferNeuronStride
                                + (size_t)wread * spikeBufferTimeStride];
            float sSpike = sbit ? 1.0f : 0.0f;
            sSpike = std::max(sSpike, spikeInputOverwriteVector[idx]);  // overwrite can raise it

            // ---- gating product (2 V, 1 Ca, 1 Spike) with int exponents ----
            float g0 = powiFast(xRow[0], (int)eRow[0]);
            float g1 = powiFast(xRow[1], (int)eRow[1]);
            float g2 = powiFast(xRow[2], (int)eRow[2]);
            float g3 = powiFast(xRow[3], (int)eRow[3]);
            float gatingTotal = ((g0 * g1) * (g2 * g3));

            // ---- current ----
            float g = conductance[idx] + conductanceGain[idx] * efficacy[idx];
            float I = g * gatingTotal * (V - Erevs[idx]);
            channelCurrent[idx]   = I;
            CaChannelCurrent[idx] = isThisCaCurrent[idx] ? I : 0.0f;
            debugRecording[idx]   = gatingTotal;

            // ---- plasticity ----
            float dp = calcWeightChange(efficacy[idx], Ca, rate[idx],
                                        gammaP[idx], gammaD[idx], thetaP[idx], thetaD[idx]);
            efficacy[idx] += timeStep * dp;

            // ---- gate updates (specialized 4 gates) ----
            // V gate 0
            xRow[0] = gateStep(xRow[0], V, timeStep,
                            vHalfR[0], vKiR[0], tau0R[0], tau1R[0], thR[0], tkInvR[0], minOpR[0]);
            // V gate 1
            xRow[1] = gateStep(xRow[1], V, timeStep,
                            vHalfR[1], vKiR[1], tau0R[1], tau1R[1], thR[1], tkInvR[1], minOpR[1]);
            // Ca gate
            xRow[2] = gateStep(xRow[2], Ca, timeStep,
                            vHalfR[2], vKiR[2], tau0R[2], tau1R[2], thR[2], tkInvR[2], minOpR[2]);

            xRow[3] = gateStep(xRow[3], sSpike, timeStep,
                                 vHalfR[3], vKiR[3], tau0R[3], tau1R[3], thR[3], tkInvR[3], minOpR[3]);

        }



        auto channelEnd = std::chrono::steady_clock::now();
        channelT += std::chrono::duration<double>(channelEnd - initializationEnd).count();

        #pragma omp parallel for schedule(static) // uniform work per neuron
        for (int n=0; n<nNeurons; ++n){
            float sI=0.0f, sCa=0.0f;
            int start = neuronOffsets[n], end = neuronOffsets[n+1];

            // encourage vectorization
            //#pragma omp simd reduction(+:sI,sCa)
            for (int k=start; k<end; ++k){
                int c = chanIndexByNeuron[k];
                float I = channelCurrent[c];
                sI  += I;
                sCa += I * (float)isThisCaCurrent[c];  // or use pre-stored 0.0/1.0
            }
            netCurrent[n]   += sI;
            netCaCurrent[n] += sCa;
        }

        auto accumulateEnd = std::chrono::steady_clock::now();
        accumulateT += std::chrono::duration<double>(accumulateEnd - channelEnd).count();


        //Solve diffusive connections. could also be parallelized, but its relatively not as big
        for (size_t idx=0;idx<nDiffusiveConnections;idx++){
            int preSynapticNeuron = diffPreNeuron[idx];
            int postSynapticNeuron = diffPostNeuron[idx];

            float Vpost = voltages[postSynapticNeuron];
            float Capost = calciumVals[postSynapticNeuron];
            float Vpre = voltages[preSynapticNeuron];
            float Capre = calciumVals[preSynapticNeuron];

            float I = diffRate[idx]*(Vpost - Vpre);
            float CaFlow = diffRate[idx]*(Capost - Capre);

            netCurrent[postSynapticNeuron] += I;
            netCaCurrent[postSynapticNeuron] += CaFlow*I;

            netCurrent[preSynapticNeuron] -= I;
            netCaCurrent[preSynapticNeuron] -= CaFlow*I;

        }
        auto diffusiveEnd = std::chrono::steady_clock::now();
        diffusiveT += std::chrono::duration<double>(diffusiveEnd - accumulateEnd).count();
            
        //Solve neurons
        for (size_t idx=0;idx<nNeurons;idx++){
            voltages[idx] -= timeStep*(netCurrent[idx]+biasCurrents[idx] +biasCurrentOverwriteVector[idx] ); //+bias overwrite list 
            calciumVals[idx] += timeStep*(-alpha*netCaCurrent[idx] +
            (-calciumVals[idx] + CaEquilibriumVal[idx])*CaRemovalRate[idx]);
        }
        auto neuronEnd = std::chrono::steady_clock::now();
        neuronT += std::chrono::duration<double>(neuronEnd - diffusiveEnd).count();

        // //Solve spikers
        // 
        // const bool pow2 = (maxDelayInt & (maxDelayInt - 1)) == 0;
        // const int mask  = maxDelayInt - 1;
        // #pragma omp parallel for schedule(dynamic, 512)
        // for (long long int idx=0;idx<nSpikers;idx++){
        //     float r = uniformFloatDist(gen);
        //     bool randomOutput = r<spikerProbability[idx];
        //     bool spikeOutput = spikerNeuronScale[idx] * (float)((voltages[idx]>vThreshold));
        //     spikeBuffer[idx*spikeBufferNeuronStride + writeIndex[idx]] = randomOutput || spikeOutput; //think L and R values for later 
        //     writeIndex[idx] = (writeIndex[idx]+1)%maxDelayInt;
        // }

        constexpr float vThreshold = 0.0f;
        #pragma omp parallel
        {
            // Per-thread RNG; mix in thread id and time step to avoid identical streams
            const int tid = omp_get_thread_num();
            Xoshiro32 rng(baseSeed
                          ^ (0x9E3779B97F4A7C15ull * (uint64_t)(1 + tid))
                          ^ ((uint64_t)timeInd << 32));

            #pragma omp for schedule(dynamic, 512)
            for (long long idx = 0; idx < (long long)nSpikers; ++idx) {

                // 1) Random term
                float r = rng.nextFloat();                             // [0,1)
                bool randomOutput = r < spikerProbability[idx];

                // 2) Spike term (branchless compare; keep your scale semantics)
                // If spikerNeuronScale is a 0/1 gate, you can fold this to a bit op; here we keep your multiply.
                float spikeTerm = spikerNeuronScale[idx] * (float)(voltages[idx] > vThreshold);
                bool spikeOutput = (spikeTerm != 0.0f);

                // 3) Write into ring buffer (use both strides to respect NumPy layout)
                int w = writeIndex[idx];                               // current head
                const size_t base = (size_t)idx * spikeBufferNeuronStride;
                spikeBuffer[base + (size_t)w * spikeBufferTimeStride] = (randomOutput || spikeOutput);

                // 4) Advance head with mask (power-of-two fast path)
                writeIndex[idx] = (w + 1) & mask;
            }
        }




        auto spikerEnd = std::chrono::steady_clock::now();
        spikerT += std::chrono::duration<double>(spikerEnd - neuronEnd).count();

        //maybe add a guard for the case where there is only one sim iter, or deal with that on the python side
        if (timeInd % samplingInterval == 0) {
            //Warning: AI generated
            for (auto& [keyHandle, recObj] : recordingInfo) {
                py::handle key = keyHandle;

                auto simArr = systemInfo[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                auto recArr = recordingInfo[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

                auto simReq = simArr.request();
                auto recReq = recArr.request();

                if (recReq.ndim != simReq.ndim + 1)
                    throw std::runtime_error("Array dimension mismatch");

                for (size_t i = 0; i < simReq.ndim; ++i) {
                    if (recReq.shape[i + 1] != simReq.shape[i])
                        throw std::runtime_error("Shape mismatch between simDict and rDict");
                }

                size_t innerSize = 1;
                for (size_t i = 0; i < simReq.ndim; ++i)
                    innerSize *= static_cast<size_t>(simReq.shape[i]);

                size_t offsetElems = static_cast<size_t>(sampleInd) * innerSize;
                float* recPtr = static_cast<float*>(recReq.ptr);
                float* simPtr = static_cast<float*>(simReq.ptr);

                std::memcpy(recPtr + offsetElems, simPtr, innerSize * sizeof(float));
            }
            sampleInd++;
            auto recordingEnd = std::chrono::steady_clock::now();
            recordingT += std::chrono::duration<double>(recordingEnd - spikerEnd).count();
        }
        

        
    }

   
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    py::print("Elapsed time in seconds:", elapsed.count());
    
    // py::print("Initialization time:", initializationT);
    // py::print("Channel solving time:", channelT);
    // py::print("Accumulation time:", accumulateT);
    // py::print("Diffusive connections time:", diffusiveT);
    // py::print("Neuron solving time:", neuronT);
    // py::print("Spiker solving time:", spikerT);
    // py::print("Recording time:", recordingT);
    // py::print("Total time:", initializationT + channelT + accumulateT + diffusiveT + neuronT + spikerT + recordingT);

    performanceStats[0] = initializationT;
    performanceStats[1] = channelT;
    performanceStats[2] = accumulateT;
    performanceStats[3] = diffusiveT;
    performanceStats[4] = neuronT;
    performanceStats[5] = spikerT;
    performanceStats[6] = recordingT;
    performanceStats[7] = initializationT + channelT + accumulateT + diffusiveT + neuronT + spikerT + recordingT;

    return py::none();
}

py::object testFunc(py::array_t<int> testIntVal,
                    py::array_t<double> testNParr) 
    {
        int testInt = *testIntVal.data();
        auto doubArr = (double *) testNParr.request().ptr;
        doubArr[testInt] = 2.5;

        return py::cast<py::none>(Py_None);
    }



        // for (int idx=0;idx<nChannels;idx++){
        //     //temporary variables
        //     postSynapticNeuron = postNeuron[idx];
        //     V = voltages[postSynapticNeuron];
        //     Ca = calciumVals[postSynapticNeuron];
        //     preSynapticNeuron = channelPreSpiker[idx];
        //     //spikes = spikeVals[postSynapticNeuron];

        //     g = conductance[idx] + conductanceGain[idx]*efficacy[idx];

        //     //multiply gating variables to get net gate opening
        //     gatingTotal = 1.0f;
        //     for (size_t idy=0;idy<nGatingVariables;idy++){
        //         float expnt = exponent[idx*chanStride + idy];
        //         gatingTotal *= pow(x[idx*chanStride + idy],expnt);
        //     }

        //     I = g*gatingTotal*(V-Erevs[idx]);// g*gatingTotal*(V-Erev);
        //     // netCurrent[postSynapticNeuron] += I;
        //     // netCaCurrent[postSynapticNeuron] += isThisCaCurrent[idx]*I;
        //     channelCurrent[idx] = I; //we accumulate later
        //     CaChannelCurrent[idx] = isThisCaCurrent[idx]*I;
        //     debugRecording[idx] = gatingTotal;

        //     readIndex = ((writeIndex[preSynapticNeuron] - inputDelay[idx]) % maxDelayInt + maxDelayInt) % maxDelayInt;
        //     //readIndex = (writeIndex[preSynapticNeuron]-inputDelay[idx])%maxDelayInt;
        //     perceivedSpike = (float)spikeBuffer[preSynapticNeuron*spikeBufferNeuronStride+readIndex]; //shit its a flattened 2d array, need to deal with the awkward indexing 
        //     perceivedSpike = std::max(perceivedSpike,spikeInputOverwriteVector[idx]); //stimulateion 

        //     float dp = calcWeightChange(efficacy[idx],Ca,rate[idx],gammaP[idx],gammaD[idx],thetaP[idx],thetaD[idx]);
        //     efficacy[idx] +=timeStep*dp;

        //     for (size_t idy=0;idy<nGatingVariables;idy++) {
        //         if (idy<2) gateVar = V;
        //         else if (idy==2) gateVar = Ca; //Ca
        //         else gateVar = perceivedSpike; //Spikes

        //         float xp = x[idx*chanStride + idy];
        //         float vHalfp = vHalf[idx*chanStride + idy];
        //         float vKip = vKi[idx*chanStride + idy];
        //         float tau0p = tau0[idx*chanStride + idy];
        //         float tau1p = tau1[idx*chanStride + idy];
        //         float thp = th[idx*chanStride + idy];
        //         float tkInvp = tkInv[idx*chanStride + idy];
        //         float minOp = minOpenness[idx*chanStride + idy];

        //         x[idx*chanStride + idy] = solveGate(xp,gateVar,timeStep,vHalfp,vKip,tau0p,tau1p,thp,tkInvp,minOp);


        //     }

        // }
    // auto arr = py::array_t<float>({2, 3});
    // recordingInfo["test"] = arr;
    // auto buf = arr.mutable_unchecked<2>();
    // buf(0, 0) = 42.0f;
    // py::print(recordingInfo["test"]);


    // for (auto item : recordingInfo) {
    //         //Warning: AI generated 
    //         py::handle key = item.first;
    //         py::print(key);
    //         auto recBuf = recordingInfo[key].cast<py::array_t<float>>().request();
    //         auto recArray = (float*)recBuf.ptr;
    //         py::print("recordingInfo[key] is:", recordingInfo[key].get_type());
    //         size_t length = recBuf.size;
    //         py::print("After:", recordingInfo[key]);
    //     }