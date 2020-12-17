from math import floor
import random

class line_location():
    timestep = 0.01

    def __init__(self,worldSize=1,senderPos=None,receiverPos=None,goal=None):
        self.worldSize = worldSize
        if goal == None:
            goal = random.uniform(0.5*worldSize,worldSize)
        if senderPos == None:
            senderPos = random.uniform(0,0.3*worldSize)
        if receiverPos == None:
            receiverPos = random.uniform(0,0.3*worldSize)
        self.goal = goal
        self.senderPos = senderPos
        self.receiverPos = receiverPos
        self.t = 0

    # Currently a simple version with no acceleration
    def step(self, senderdx=0, receieverdx=0):
        self.t += self.timestep
        
        # Update Pos
        self.updatePos(True,senderdx)
        self.updatePos(False,receieverdx)

    def updatePos(self,isSender,dx):
        pos = self.senderPos if isSender else self.receiverPos
        pos += dx

        # There actually isn't a restriction on bounds, wow!
        # if pos < 0:
        #     pos = 0
        # elif pos > self.worldSize:
        #     pos = self.worldSize
        
        if isSender:
            # This is where additional checks can be performed on the senders movement
            self.senderPos = pos
        else:
            self.receiverPos = pos


    # Return the sensors, that is the Position, Contact and Target Position sensors
    def getState(self,isSender):
        contactSensor = 1 if abs(self.senderPos - self.receiverPos) <= 0.04 else 0
        if isSender:
            targetSensor = abs(self.senderPos - self.goal)
            return [contactSensor,self.senderPos,targetSensor]
        else:
            return [contactSensor,self.receiverPos,-1]

    
    def getAsciiState(self):
        state = ['-'] * 101 #Positions from 0 to 100
        if 0 < self.senderPos < self.worldSize:
            state[floor((self.senderPos/self.worldSize)*100)] = 'S'
        if 0 < self.receiverPos < self.worldSize:
            state[floor((self.receiverPos/self.worldSize)*100)] = 'R'
        state[floor((self.goal/self.worldSize)*100)] = '#'
        return f"t = {self.t:.2f}\n" + ''.join(state)

    # linear scale, high when receiver is close to goal
    def fitness(self):
        # print(f"Receiver at {self.receiverPos}, fitness {max(self.worldSize - abs(self.receiverPos-self.goal),0)}")
        return max(self.worldSize - abs(self.receiverPos-self.goal),0)

    # motor must always go
    # make this contnuous
    def motor(self,val):
        if val < 0.25:
            return -self.worldSize/100
        elif val > 0.75: 
            return self.worldSize/100
        else:
            return 0

# Testing, 1 second movement
def main():
    import time
    sim = line_location(100)
    for i in range(100):
        moves = random.choices([-1,-2,0,1,2],k = 2)
        sim.step(*moves)
        print(sim.getAsciiState())
        # time.sleep(0.01)
    print(f"Fitness is {sim.fitness()}\nSender has sensors {sim.getState(True)}\nReceiver has sensors {sim.getState(False)}")

if __name__ == '__main__':
    main()
    