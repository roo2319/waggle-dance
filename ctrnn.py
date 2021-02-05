import numpy as np
from scipy.special import expit

def sigmoid(x):
    return 1/(1 + np.exp(-x))



class CTRNN():
    def __init__(self, genome=None):
        if genome == None:
            genome = Genome()
        self.inputsCount   = genome.inputsCount
        self.outputCount   = genome.outputsCount
        self.hiddenCount   = genome.hiddenCount
        self.inputWeights  = np.pad(genome.inputWeights,(0,self.hiddenCount - len(genome.inputWeights)))
        self.outputWeights = np.pad(genome.outputWeights,(0,self.hiddenCount - len(genome.outputWeights)))
        self.weights       = genome.weights
        self.biases        = genome.biases
        self.gains         = genome.gains
        self.states        = np.zeros((genome.hiddenCount))
        self.outputs       = expit(np.multiply(self.gains,(self.states + self.biases)))
        self.rTaus         = genome.rTaus

    def eulerStep(self,externalInputs,stepsize):
        externalInputs = np.pad(externalInputs,(0,self.hiddenCount-len(externalInputs)))
        # First we calculate the change for each hidden node based on external inputs
        delta = np.multiply(externalInputs, self.inputWeights) + np.matmul(self.outputs, self.weights)
        # Then we update the state of each hidden node
        self.states += np.multiply(stepsize * self.rTaus, (delta - self.states))
        # Lastly we can update the outputs of each hidden node
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))
        # We can now calculate the external output. 
        return np.multiply(self.outputWeights,self.outputs)[:self.outputCount]
    
    def reset(self):
        self.states  = np.zeros(self.states.shape)
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))

    def randomizeStates(self,low,high):
        self.states  = np.random.uniform(low,high,self.states.shape)
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))
        

class Genome():
    def __init__(self,inputsCount=3,hiddenCount=3,outputsCount=1,iWeights=None,oWeights=None,weights=None,
                biases=None, gains=None, taus=None):
        # Weights = 2D array 
        self.inputsCount  = inputsCount
        self.hiddenCount  = hiddenCount
        self.outputsCount = outputsCount
        self.size = inputsCount + hiddenCount + outputsCount

        # Each input/output is connected to exactly one hidden node
        if iWeights is None:
            # iWeights = np.random.normal(size=(inputsCount))
            iWeights = np.zeros(inputsCount)
        self.inputWeights = iWeights
        
        if oWeights is None:
            # oWeights = np.random.normal(size=(outputsCount))
            oWeights = np.zeros(outputsCount)
        self.outputWeights = oWeights

        if weights is None:
            # weights = np.random.normal(size=(hiddenCount,hiddenCount))
            weights = np.zeros((hiddenCount,hiddenCount))
        self.weights = weights

        if biases is None: 
            # biases = np.random.normal(size=(hiddenCount)) 
            biases = np.zeros(hiddenCount)
        self.biases = biases

        if gains is None:
            # gains = np.random.normal(size=(hidden))
            gains = np.ones((hiddenCount))
        self.gains = gains

        if taus is None:
            taus = np.ones((hiddenCount))
        self.taus = taus
        self.rTaus = np.reciprocal(self.taus)


    # Apply a gaussian mutation of 0.2 to every parameter
    # Potentially this could be multiplicative
    def mutate(self):
        self.inputWeights  += np.random.normal(0,0.2,self.inputWeights.shape)
        self.outputWeights += np.random.normal(0,0.2,self.outputWeights.shape)
        self.weights       += np.random.normal(0,0.2,self.weights.shape)
        self.biases        += np.random.normal(0,0.2,self.biases.shape)
        self.gains         += np.random.normal(0,0.2,self.gains.shape)
        self.taus          += np.random.normal(0,0.2,self.taus.shape)

        
        self.inputWeights   = np.clip(self.inputWeights,-16,16)
        self.outputWeights  = np.clip(self.outputWeights,-16,16)
        self.weights        = np.clip(self.weights,-16,16)
        self.biases         = np.clip(self.biases,-16,16)
        self.gains          = np.clip(self.gains,-10,10)
        self.taus           = np.clip(self.taus,1,100)

    def copy(self):
        return Genome(self.inputsCount, self.hiddenCount, self.outputsCount, np.copy(self.inputWeights),
                      np.copy(self.outputWeights), np.copy(self.weights), np.copy(self.biases),
                      np.copy(self.gains), np.copy(self.taus))

if __name__ == '__main__':
    # Simple oscillator example, taken from Randall Beer
    genome = Genome(0,2,0,[],[],np.array([[4.5,-1],[1,4.5]]),[-2.75,-1.75],None,None)
    nn     = CTRNN(genome)
    nn.randomizeStates(-0.5,0.5)
    # exit()
    time = 0
    while time < 250:
        nn.eulerStep([],0.01)
        time += 0.01
        print(nn.outputs)

    