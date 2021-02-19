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
        """
        Advance the CTRNN by (stepsize), setting the inputs to the network to
        externalInputs

        Parameters:
            externalInputs (List(float)) : The inputs to the network at this timestep   
            stepsize (float) : The change in time being calculated
        
        Return:
            The outputs of the network at this timestep
        """
        # externalInputs = np.pad(externalInputs,(0,self.hiddenCount-len(externalInputs)))
        paddedInputs = np.zeros(self.hiddenCount)
        paddedInputs[:self.inputsCount] = externalInputs
        # First we calculate the change for each hidden node based on external inputs
        delta = np.multiply(paddedInputs, self.inputWeights) + np.matmul(self.outputs, self.weights)
        # Then we update the state of each hidden node
        self.states += np.multiply(stepsize * self.rTaus, (delta - self.states))
        # Lastly we can update the outputs of each hidden node
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))
        # We can now calculate the external output. 
        return np.multiply(self.outputWeights,self.outputs)[:self.outputCount]
    
    def reset(self):
        """
        Set the internal states of the network to 0
        """
        self.states  = np.zeros(self.states.shape)
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))

    def randomizeStates(self,low,high):
        """
        Set the internal states of the network to random values between low and high

        Parameters: 
            low (float) 
            high (float)
        """
        self.states  = np.random.uniform(low,high,self.states.shape)
        self.outputs = expit(np.multiply(self.gains,(self.states + self.biases)))
        

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
        oWeights (np.array(float)) : A 1d list of weights from each hidden node to 
                                     it's connected output node, length = outputsCount
        weights (np.array(float))  : A 2d array of weights between each hidden node,
                                     shape = (hiddenCount,hiddenCount)
        biases (np.array(float))   : The biases of each hidden node, length = hiddenCount
        gains (np.array(float))    : The gains of each hidden node, length = hiddenCount
        taus  (np.array(float))    : The time constants of each hidden node, length = hiddenCount
        centerCrossing (bool)      : Determine if the CTRNN should be center crossing

    """
    def __init__(self,inputsCount=3,hiddenCount=3,outputsCount=1,iWeights=None,oWeights=None,weights=None,
                biases=None, gains=None, taus=None, centerCrossing = False):
        # Weights = 2D array 
        self.inputsCount  = inputsCount
        self.hiddenCount  = hiddenCount
        self.outputsCount = outputsCount
        self.size = inputsCount + hiddenCount + outputsCount

        # Each input/output is connected to exactly one hidden node
        if iWeights is None:
            iWeights = np.random.normal(scale=2,size=(inputsCount))
            # iWeights = np.random.uniform(-16,16,size=(inputsCount))
            # iWeights = np.zeros(inputsCount)
        self.inputWeights = iWeights
        
        if oWeights is None:
            oWeights = np.random.normal(scale=2,size=(outputsCount))
            # oWeights = np.random.uniform(-16,16,size=(outputsCount))
            # oWeights = np.zeros(outputsCount)
        self.outputWeights = oWeights

        if weights is None:
            weights = np.random.normal(scale=2,size=(hiddenCount,hiddenCount))
            # weights = np.random.uniform(-16,16,size=(hiddenCount,hiddenCount))

        self.weights = weights

        if centerCrossing == True:
            biases = -0.5 * sum(weights)
        elif biases is None: 
            biases = np.random.normal(scale=2,size=(hiddenCount)) 
            # biases = np.random.uniform(-16,16,size=(hiddenCount))
            # biases = np.ones(hiddenCount)

        self.biases = biases

        if gains is None:
            gains = np.random.normal(scale=2,size=(hiddenCount))
            # gains = np.random.uniform(-25,25,size=(hiddenCount))
            # gains = np.ones((hiddenCount))

        self.gains = gains

        if taus is None:
            taus = np.ones((hiddenCount))
            # taus = np.random.uniform(50,100,(hiddenCount))
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
        self.gains         += np.random.normal(0,stddev,self.gains.shape)
        self.taus          += np.random.normal(0,stddev,self.taus.shape)

        
        # self.inputWeights   = np.clip(self.inputWeights,-16,16)
        # self.outputWeights  = np.clip(self.outputWeights,-16,16)
        # self.weights        = np.clip(self.weights,-16,16)
        # self.biases         = np.clip(self.biases,-16,16)
        # self.gains          = np.clip(self.gains,-10,10)
        self.taus           = np.clip(self.taus,1,100)
        self.rTaus          = np.reciprocal(self.taus)

    def beerMutate(self, stddev):
        """
        Perform mutation as described by Randall Beer
        """
        magnitude = np.random.normal(0,stddev)
        mutationvector = np.random.normal(0,1,self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)+(3*self.hiddenCount))
        mutationvector /= np.sqrt((mutationvector**2).sum(-1))
        mutationvector *= magnitude

        self.inputWeights  += mutationvector[:self.inputsCount]
        self.outputWeights += mutationvector[self.inputsCount:self.inputsCount+self.outputsCount]
        self.weights       += mutationvector[self.inputsCount+self.outputsCount:self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)].reshape((self.hiddenCount,self.hiddenCount))
        self.biases        += mutationvector[self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount):self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)+self.hiddenCount]
        self.gains         += mutationvector[self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)+self.hiddenCount:self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)+(2*self.hiddenCount)]
        self.taus          += mutationvector[self.inputsCount+self.outputsCount+(self.hiddenCount*self.hiddenCount)+(2*self.hiddenCount):]


    def copy(self):
        """
        A method to create a copy of a genome object

        Return:
            A genome object that is identical to the caller
        """
        return Genome(inputsCount=self.inputsCount, hiddenCount=self.hiddenCount, outputsCount=self.outputsCount, iWeights=np.copy(self.inputWeights),
                      oWeights=np.copy(self.outputWeights), weights=np.copy(self.weights), biases=np.copy(self.biases),
                      gains=np.copy(self.gains), taus=np.copy(self.taus))
    
    def __str__(self):
        """
        A string magic method to convert the object to a printable representation

        Return:
            String
        """
        return f"Input Weights: {self.inputWeights} \nOutput weights: {self.outputWeights}\n" +\
               f"Weights: {self.weights} \nBiases: {self.biases} \n Gains: {self.gains} \n Taus: {self.taus}"
        

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

    