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

class Point():
    
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self,other):
        return Point(self.x+other.x,self.y+other.y,self.z+other.z)

    def clip(self,r):
        self.x = quickClip(-r,r,self.x)
        self.y = quickClip(-r,r,self.y)
        self.z = quickClip(-r,r,self.z)

    # Component wise distance
    def compdist(self,other):
        return Point(other.x-self.x,other.y-self.y,other.z-self.z)

    # Euclidean distance
    def dist(self,other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)

    def toList(self):
        return [self.x,self.y,self.z]

    def __str__(self):
        return f"({self.x},{self.y},{self.z})"
    
    def __eq__(self, other):
        if other == None:
            return False
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z) 



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
    return (2 * (val - 0.5))  * 0.01 * cube_location.timestep

motors = {"discreteMotor":discreteMotor, "clippedMotor1" : clippedMotor1, "clippedMotor2" : clippedMotor2, "clippedMotor3" : clippedMotor3, "sigmoidMotor" : sigmoidMotor, "tanhMotor" : tanhMotor, "camposMotor":camposMotor}
motorFunction = clippedMotor1

class cube_location():
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
            goal = Point(*[random.uniform(0.5,1),random.uniform(0.5,1),random.uniform(0.5,1)])
        if senderPos == None:
            senderPos = Point(*[random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)])
        if receiverPos == None:
            receiverPos = Point(*[random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)])
        self.goal = goal
        self.senderPos = senderPos
        self.receiverPos = receiverPos
        self.t = 0
        self.ctime = 0
        self.touches = 0
        self.prevcon = 1

    def step(self, senderOutput=[0,0,0], receieverOutput=[0,0,0]):
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
        delta = Point(*map(motorFunction,output))
        pos += delta
        
        if isSender:
            # This is where additional checks can be performed on the senders movement
            # self.senderPos = pos
            pos.clip(0.6)
            self.senderPos = pos

        else:
            self.receiverPos = pos


    def getState(self,isSender):
        """
        Gather the sensor values for a given agent, based on whether or not they 
        are the sender
    
        Parameters:
            isSender (bool) : Return the senders sensor values if this is true

        Return:
            A list(float) containing the three distance sensors and the three target sensors
        """
        if isSender:
            a  = self.senderPos
            oa = self.receiverPos 
        else:
            a  = self.receiverPos
            oa = self.senderPos 

        OASens = a.compdist(oa)
        
        if isSender:
            targetSensor = a.compdist(self.goal)
            return  OASens.toList() + targetSensor.toList()
        else:
            return  OASens.toList() + [-1,-1,-1]

    
    def getLoggingData(self):
        """
        Returns data that can be useful for logging

        Return: 
        A tuple containing the positions of both agents and the current value for T
        """
        return (self.receiverPos, self.senderPos, self.t)


    def fitness(self):
        """
        A fitness function, with a higher value when the receiver is close to the goal
        The max value is 1, the minimum value is 0.

        Return:
            The fitness of the simulation (float)
        """
        # Receiver goal
        return max(1 - self.receiverPos.dist(self.goal),0)


# Testing, 1 second movement
def main():
    sim = cube_location()
    print(f"Fitness is {sim.fitness()}\nSender has sensors {sim.getState(True)}\nReceiver has sensors {sim.getState(False)}")

if __name__ == '__main__':
    main()
    