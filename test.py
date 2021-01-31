import pickle
import random
import sys
import line_location
import ctrnn

# load the winner
if len(sys.argv) < 2:
    print("Usage: py ./test.py genome_name.pkl")
    exit()

with open(sys.argv[1], 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)



positions = (random.uniform(0,0.3),random.uniform(0,0.3))
sp,rp = positions
print("Initial conditions:")
print("   Sender = {0:.4f}".format(sp))
print(" Receiver = {0:.4f}".format(rp))
print()
for goal in [0.5,0.6,0.7,0.8,0.9]:
    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
    time_const = line_location.line_location.timestep


    sender = ctrnn.CTRNN(c)
    receiver = ctrnn.CTRNN(c)

    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    while sim.t < 3:
        act1 = sender.eulerStep(sim.getState(True),time_const)
        act2 = receiver.eulerStep(sim.getState(False),time_const)
        senderdx = sim.motor(act1[0])
        receiverdx = sim.motor(act2[0])
        # We can model force here
        # print(f"network thinks {act1,act2}")
        # We can model force here
        sim.step(senderdx,receiverdx)
        # print(sim.getAsciiState())




    print("Final conditions:")
    print("   Sender = {0:.4f}".format(sim.senderPos))
    print(" Receiver = {0:.4f}".format(sim.receiverPos))
    print("     Goal = {0:.4f}".format(sim.goal))
    print(f" fitness = {sim.fitness()}")
    print()

