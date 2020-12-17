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
        self.states = np.multiply(stepsize * self.rTaus, (delta - self.states))
        # Lastly we can update the outputs of each hidden node
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))
        
        # We can now calculate the external output. 
        return np.multiply(self.outputWeights,self.outputs)[:self.outputCount]
        # print(delta)
        # print(self.states)
        # print(self.outputs)
    
    def reset(self):
        self.states  = np.zeros(self.states.shape)
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))

        

class Genome():
    def __init__(self,inputs=3,hidden=3,outputs=1,iWeights=None,oWeights=None,weights=None,
                biases=None, gains=None, taus=None):
        # Weights = 2D array 
        self.inputsCount  = inputs
        self.hiddenCount  = hidden
        self.outputsCount = outputs
        self.size = inputs + hidden + outputs

        # Each input/output is connected to exactly one hidden node
        if iWeights is None:
            iWeights = np.random.normal(size=(inputs))
        self.inputWeights = iWeights
        
        if oWeights is None:
            oWeights = np.random.normal(size=(outputs))
        self.outputWeights = oWeights

        if weights is None:
            weights = np.random.normal(size=(hidden,hidden))
        self.weights = weights

        if biases is None: 
            biases = np.random.normal(size=(hidden)) 
        self.biases = biases

        if gains is None:
            gains = np.random.normal(size=(hidden))
        self.gains = gains

        if taus is None:
            taus = np.random.normal(size=(hidden))
        self.taus = taus
        self.rTaus = np.reciprocal(self.taus)


    # Apply a gaussian mutation of 0.2 to every parameter
    def mutate(self):
        self.inputWeights  += np.random.normal(0,0.2,self.inputWeights.shape)
        self.outputWeights += np.random.normal(0,0.2,self.outputWeights.shape)
        self.weights       += np.random.normal(0,0.2,self.weights.shape)
        self.biases        += np.random.normal(0,0.2,self.biases.shape)
        self.gains         += np.random.normal(0,0.2,self.gains.shape)
        self.taus          += np.random.normal(0,0.2,self.taus.shape)

    def copy(self):
        return Genome(self.inputsCount, self.hiddenCount, self.outputsCount, np.copy(self.inputWeights),
                      np.copy(self.outputWeights), np.copy(self.weights), np.copy(self.biases),
                      np.copy(self.gains), np.copy(self.taus))

if __name__ == '__main__':
    nn = CTRNN()

    