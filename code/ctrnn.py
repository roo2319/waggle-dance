import numpy as np
from scipy.special import expit

def sigmoid(x):
    return 1/(1 + np.exp(-x))



class CTRNN():
    """
    Class implementing Continuous Time Recurrent Neural Networks

    Parameters:
        genome (Genome)
    """
    
    def __init__(self, genome=None,timestep=1):
        if genome == None:
            genome = Genome()
        self.inputsCount   = genome.inputsCount
        self.outputCount   = genome.outputsCount
        self.hiddenCount   = genome.hiddenCount
        self.inputWeights  = np.pad(genome.inputWeights,(0,self.hiddenCount - len(genome.inputWeights)))
        self.paddedInputs = np.zeros(self.hiddenCount)
        self.weights       = genome.weights
        self.biases        = genome.biases
        self.states        = np.zeros((genome.hiddenCount))
        self.outputs       = expit((self.states + self.biases))
        self.rTaus         = genome.rTaus
        self.sTaus         = self.rTaus * timestep

    def eulerStep(self,externalInputs):
        """
        Advance the CTRNN by (stepsize), setting the inputs to the network to
        externalInputs

        Parameters:
            externalInputs (List(float)) : The inputs to the network at this timestep   
            stepsize (float) : The change in time being calculated
        
        Return:
            The outputs of the network at this timestep
        """
        self.paddedInputs[:self.inputsCount] = externalInputs

        # First we calculate the change for each hidden node based on external inputs
        delta = np.multiply(self.paddedInputs, self.inputWeights) + np.dot(self.outputs, self.weights) 

        # Then we update the state of each hidden node
        self.states += np.multiply(self.sTaus, (delta - self.states)) 

        # We can now calculate the external output. 
        self.outputs = expit((self.states + self.biases)) 
        return self.outputs[:self.outputCount] 
    
    def reset(self):
        """
        Set the internal states of the network to 0
        """
        self.states  = np.zeros(self.states.shape)
        self.outputs = expit((self.states + self.biases))

    def randomizeStates(self,low,high):
        """
        Set the internal states of the network to random values between low and high

        Parameters: 
            low (float) 
            high (float)
        """
        self.states  = np.random.uniform(low,high,self.states.shape)
        self.outputs = expit((self.states + self.biases))
        

class Genome():
    """
    The genotype of a CTRNN. This will be the target of the genetic algorithm.
    For these networks all of the hidden nodes are fully connected to eachother, 
    and can connect to up to one input node and up to one output node

    Parameters:
        inputsCount (int) : The number of external inputs to the network, < hiddenCount
        hiddenCount (int) : The number of hidden nodes in the network
        outputsCount (int) : The number of output nodes in the network, < hiddenCount
        iWeights (np.array(float)) : A 1d list of weights from each input node to 
                                     a hidden nodelength = inputsCount
        weights (np.array(float))  : A 2d array of weights between each hidden node,
                                     shape = (hiddenCount,hiddenCount)
        biases (np.array(float))   : The biases of each hidden node, length = hiddenCount
        taus  (np.array(float))    : The time constants of each hidden node, length = hiddenCount
        centerCrossing (bool)      : Determine if the CTRNN should be center crossing

    """
    def __init__(self,inputsCount=3,hiddenCount=3,outputsCount=1,iWeights=None,weights=None,
                biases=None, taus=None, centerCrossing = False):
        # Weights = 2D array 
        self.inputsCount  = inputsCount
        self.hiddenCount  = hiddenCount
        self.outputsCount = outputsCount
        self.size = inputsCount + hiddenCount + outputsCount

        # Each input/output is connected to exactly one hidden node
        if iWeights is None:
            # iWeights = np.random.normal(scale=2,size=(inputsCount))
            iWeights = np.random.uniform(-16,16,size=(inputsCount))
            # iWeights = np.zeros(inputsCount)
        self.inputWeights = iWeights
        


        if weights is None:
            # weights = np.random.normal(scale=2,size=(hiddenCount,hiddenCount))
            weights = np.random.uniform(-16,16,size=(hiddenCount,hiddenCount))

        self.weights = weights

        if centerCrossing == True:
            biases = -0.5 * sum(weights)
        elif biases is None: 
            # biases = np.random.normal(scale=2,size=(hiddenCount)) 
            biases = np.random.uniform(-16,16,size=(hiddenCount))
            # biases = np.ones(hiddenCount)

        self.biases = biases

        if taus is None:
            # taus = np.ones((hiddenCount))
            taus = np.random.uniform(50,100,(hiddenCount))
        self.taus = taus
        self.rTaus = np.reciprocal(self.taus)


    # Recreate the beer version
    def mutate(self, stddev):
        """
        A method to mutate a genome by adding gaussian noise. 

        Parameters:
            stddev (float) : The standard deviation of the gaussian distribution to be drawn upon
        """
        self.inputWeights  += np.random.normal(0,stddev,self.inputWeights.shape)
        self.outputWeights += np.random.normal(0,stddev,self.outputWeights.shape)
        self.weights       += np.random.normal(0,stddev,self.weights.shape)
        self.biases        += np.random.normal(0,stddev,self.biases.shape)
        self.taus          += np.random.normal(0,stddev,self.taus.shape)

        
        self.inputWeights   = np.clip(self.inputWeights,-16,16)
        self.weights        = np.clip(self.weights,-16,16)
        self.biases         = np.clip(self.biases,-16,16)
        self.taus           = np.clip(self.taus,50,100)
        self.rTaus          = np.reciprocal(self.taus)

    def beerMutate(self, stddev):
        """
        Perform mutation as described by Randall Beer
        """
        magnitude = np.random.normal(0,stddev)
        mutationvector = np.random.normal(0,1,self.inputsCount+(self.hiddenCount*self.hiddenCount)+(2*self.hiddenCount))
        mutationvector /= np.sqrt((mutationvector**2).sum(-1))
        mutationvector *= magnitude

        self.inputWeights  += 16 * mutationvector[:self.inputsCount]
        self.weights       += 16 * mutationvector[self.inputsCount:self.inputsCount+(self.hiddenCount*self.hiddenCount)].reshape((self.hiddenCount,self.hiddenCount))
        self.biases        += 16 * mutationvector[self.inputsCount+(self.hiddenCount*self.hiddenCount):self.inputsCount+(self.hiddenCount*self.hiddenCount)+self.hiddenCount]
        self.taus          += 25 * mutationvector[self.inputsCount+(self.hiddenCount*self.hiddenCount)+(self.hiddenCount):self.inputsCount+(self.hiddenCount*self.hiddenCount)+2*(self.hiddenCount)]

        self.inputWeights   = np.clip(self.inputWeights,-16,16)
        self.weights        = np.clip(self.weights,-16,16)
        self.biases         = np.clip(self.biases,-16,16)
        self.taus           = np.clip(self.taus,50,100)
        self.rTaus          = np.reciprocal(self.taus)


    def copy(self):
        """
        A method to create a copy of a genome object

        Return:
            A genome object that is identical to the caller
        """
        return Genome(inputsCount=self.inputsCount, hiddenCount=self.hiddenCount, outputsCount=self.outputsCount, iWeights=np.copy(self.inputWeights),
                      weights=np.copy(self.weights), biases=np.copy(self.biases),
                      taus=np.copy(self.taus))
    
    def __str__(self):
        """
        A string magic method to convert the object to a printable representation

        Return:
            String
        """
        return f"Input Weights: {self.inputWeights} \n" +\
               f"Weights: {self.weights} \nBiases: {self.biases} \nTaus: {self.taus}"
        

if __name__ == '__main__':
    # Simple oscillator example, taken from Randall Beer
    genome = Genome(0,2,0,[],np.array([[4.5,-1],[1,4.5]]),[-2.75,-1.75],None,None)
    nn     = CTRNN(genome,0.01)
    nn.randomizeStates(-0.5,0.5)
    # exit()
    time = 0
    out1 = []
    out2 = []
    while time < 10000:
        nn.eulerStep([])
        time += 0.01
        outputs = nn.outputs
        out1.append(outputs[0])
        out2.append(outputs[1])

    import matplotlib.pyplot as plt
    time = np.linspace(0,10000,len(out1))
    plt.plot(time,out1)
    plt.plot(time,out2)
    plt.xlabel("Time")
    plt.ylabel("Output Value")
    plt.show()
    