from math import floor
import random
import numpy as np

class line_location():
    timestep = 0.01

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

    # Step, currently has no momentum
    def step(self, senderdx=0, receieverdx=0):
        self.t += self.timestep
        
        # Update Pos
        self.updatePos(True,senderdx)
        self.updatePos(False,receieverdx)

    def updatePos(self,isSender,dx):
        pos = self.senderPos if isSender else self.receiverPos
        pos += dx
        
        if isSender:
            # This is where additional checks can be performed on the senders movement
            # self.senderPos = pos

            self.senderPos = np.clip(pos,0,0.35)

        else:
            self.receiverPos = pos


    # Return the sensors, that is the Position, Contact and Target Position sensors
    def getState(self,isSender):
        contactSensor = 1 if abs(self.senderPos - self.receiverPos) <= 0.2 else 0
        if isSender:
            targetSensor = abs(self.senderPos - self.goal)
            return [contactSensor,self.senderPos,targetSensor]
        else:
            return [contactSensor,self.receiverPos,-1]

    
    def getLoggingData(self):
        return (self.receiverPos, self.senderPos, self.t)

    
    def getAsciiState(self):
        state = ['-'] * 101 #Positions from 0 to 100
        if 0 < self.senderPos < 1:
            state[floor(self.senderPos*100)] = 'S'
        if 0 < self.receiverPos < 1:
            state[floor(self.receiverPos*100)] = 'R'
        state[floor(self.goal*100)] = '#'
        return f"t = {self.t:.2f}\n" + ''.join(state)

    # linear scale, high when receiver is close to goal
    def fitness(self):
        # print(f"Receiver at {self.receiverPos}, fitness {max(self.worldSize - abs(self.receiverPos-self.goal),0)}")
        # Receiver goal
        return max(1 - abs(self.receiverPos-self.goal),0)
        # Sender Goal
        # return max(1 - abs(self.senderPos-self.goal),0)

    # Threshold function
    def motor(self,val):
        # if val < 0.25:
        #     return -0.01
        # elif val > 0.75: 
        #     return 0.01
        # else:
        #     return 0
        return np.clip((val-0.5)/50,-0.01,0.01)

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
    