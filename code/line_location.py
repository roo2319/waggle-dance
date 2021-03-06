from math import floor, tanh
import random
import numpy as np
from scipy.special import expit

def quickClip(minV,maxV,value):
    """
    An efficient way to clip a value to be inside a given range.
    Note that there is no check that minV < maxV but this should always be the case

    Parameters: 
        minV (float) : The minimum value
        maxV (float) : The maximum value
        value (float) : The value to be clipped

    Return:
        A float that is between minV and maxV
    """
    return max(minV,min(maxV,value))

def discreteMotor(val):
    """
    A motor function that assign -0.01, 0 or 0.01 based on a threshold

    Parameters: 
        val : The value to be transformed

    Return:
        -0.01, 0 or 0.01
    """
    if val < 0.25:
        return -0.01
    elif val > 0.75: 
        return 0.01
    else:
        return 0

def clippedMotor1(val):
    return quickClip(-0.01,0.01,(val-0.5)/50)

def clippedMotor2(val):
    return quickClip(-0.01,0.01,(val-1)/50)

def clippedMotor3(val):
    return quickClip(-0.01,0.01,val-0.59)

def sigmoidMotor(val):
    return (expit(val)-0.5)/50 

def tanhMotor(val):
    return (tanh(val)-0.5)/50

def camposMotor(val):
    return (2 * (val - 0.5))  * 0.01 * line_location.timestep

motors = {"discreteMotor":discreteMotor, "clippedMotor1" : clippedMotor1, "clippedMotor2" : clippedMotor2, "clippedMotor3" : clippedMotor3, "sigmoidMotor" : sigmoidMotor, "tanhMotor" : tanhMotor, "camposMotor":camposMotor}
motorFunction = clippedMotor1
class line_location():
    """
    The class representing the 1D agent game. The task has two agents, a sender and 
    a receiver. The goal is for the sender to communicate the location of the goal
    position to the receiver, and the receiver to navigate there.

    Parameters:
        senderPos (float) : The starting location of the sender
        receiverPos (float) : The starting location of the receiver
        goal (float) : The location of the endpoint. 
    """
    timestep = 1

    def __init__(self,senderPos=None,receiverPos=None,goal=None):
        if goal == None:
            goal = random.uniform(0.5,1)
        if senderPos == None:
            senderPos = random.uniform(0,0.3)
        if receiverPos == None:
            receiverPos = random.uniform(0,0.3)
        self.goal = goal
        self.senderPos = senderPos
        self.receiverPos = receiverPos
        self.t = 0
        self.ctime = 0
        self.touches = 0
        self.prevcon = 1

    def step(self, senderOutput=0, receieverOutput=0):
        """
        The step function, responsible for moving the game ahead a single timestep.
        This function will update positions of agents as well as increase the value for 
        time.

        Parameters:
            senderdx (float) : The change in position for the sender
            receiverdx (float) : The change in positon for the receiver
        """
        self.t += self.timestep
        
        # Update Pos
        self.updatePos(True,senderOutput)
        self.updatePos(False,receieverOutput)


    def updatePos(self,isSender,output):
        """
        This function will update the position of one of the agents, specified by
        the isSender parameter, taking in the output of that agent and applying a 
        motor function to it.

        Parameters:
            isSender (bool) : True if the agent being updated is the sender
            output (float) : The raw output of the CTRNN, passed into a motor function
        """
        pos = self.senderPos if isSender else self.receiverPos
        dx = motorFunction(output)
        pos += dx
        
        if isSender:
            # This is where additional checks can be performed on the senders movement
            # self.senderPos = pos
            
            self.senderPos = quickClip(0,0.3,pos)

        else:
            self.receiverPos = pos


    def getState(self,isSender):
        """
        Gather the sensor values for a given agent, based on whether or not they 
        are the sender
    
        Parameters:
            isSender (bool) : Return the senders sensor values if this is true

        Return:
            A list(float) containing the Contact sensor, Self Position sensor
            and Target Sensor / Constant Value Sensor
        """
        contactSensor = 1 if abs(self.senderPos - self.receiverPos) <= 0.4 else 0
        if self.prevcon != contactSensor:
            self.prevcon = contactSensor
            if self.prevcon:
                self.touches += 1
        
        if contactSensor and self.t > 150:
            self.ctime += 1 # Remember this will be counted twice
        if isSender:
            targetSensor = abs(self.senderPos - self.goal)
            return [contactSensor,self.senderPos,targetSensor]
        else:
            return [contactSensor,self.receiverPos,-1]

    
    def getLoggingData(self):
        """
        Returns data that can be useful for logging

        Return: 
        A tuple containing the positions of both agents and the current value for T
        """
        return (self.receiverPos, self.senderPos, self.t)

    
    def getAsciiState(self):
        state = ['-'] * 101 #Positions from 0 to 100
        if 0 < self.senderPos < 1:
            state[floor(self.senderPos*100)] = 'S'
        if 0 < self.receiverPos < 1:
            state[floor(self.receiverPos*100)] = 'R'
        state[floor(self.goal*100)] = '#'
        return f"t = {self.t:.2f}\n" + ''.join(state)

    def fitness(self):
        """
        A fitness function, with a higher value when the receiver is close to the goal
        The max value is 1, the minimum value is 0.

        Return:
            The fitness of the simulation (float)
        """
        # Receiver goal
        return max(1 - abs(self.receiverPos-self.goal),0)
        # Sender Goal
        # return max(1 - abs(self.senderPos-self.goal),0)
        # Touch Less
        # return max(1 - abs(self.receiverPos-self.goal) - self.touches/10,0)
        # Eval Fitness
        # return 1 if abs(self.receiverPos-self.goal) <= 0.05 else 0
        # Ctime Fitness
        # return max(1 - abs(self.receiverPos-self.goal) - self.ctime/300,0)
        # Nudge fitness
        # dist = abs(self.receiverPos-self.goal)
        # if dist <= 0.05:
        #     return 1
        # elif dist <= 0.1:
        #     return 0.5
        # elif dist <= 0.2: 
        #     return 0.1
        # else:
        #     return 0

# Testing, 1 second movement
def main():
    sim = line_location(1)
    for i in range(100):
        moves = random.choices([-0.01,-0.02,0,0.01,0.02],k = 2)
        sim.step(*moves)
        print(sim.getAsciiState())
        print(sim.getState(True))
    print(f"Fitness is {sim.fitness()}\nSender has sensors {sim.getState(True)}\nReceiver has sensors {sim.getState(False)}")

if __name__ == '__main__':
    main()
    