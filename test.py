import pickle
import random
import sys
import matplotlib.pyplot as plt
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
# positions = (0.3,0.05)

sp,rp = positions
print("Initial conditions:")
print("   Sender = {0:.4f}".format(sp))
print(" Receiver = {0:.4f}".format(rp))
print()
fig, ax = plt.subplots(3,5)
i = 0
for goal in [0.5,0.6,0.7,0.8,0.9]:
    displacement = [[],[],[]]
    rsens = [[],[],[]]
    ssens = [[],[],[]]
    sim = line_location.line_location(senderPos=sp,receiverPos=rp,goal=goal)
    time_const = line_location.line_location.timestep


    sender = ctrnn.CTRNN(c)
    receiver = ctrnn.CTRNN(c)

    sender.reset()
    receiver.reset()
    # Run the given simulation for up to num_steps time steps.
    while sim.t < 3:
        senderstate = sim.getState(True)
        receiverstate = sim.getState(False)
        act1 = sender.eulerStep(senderstate,time_const)
        act2 = receiver.eulerStep(receiverstate,time_const)

        senderOut = act1[0]
        receiverOut = act2[0]
        # We can model force here
        # print(f"network thinks {act1,act2}")
        # We can model force here
        sim.step(senderOut,receiverOut)
        r,s,t = sim.getLoggingData()

        displacement[0].append(t)
        displacement[1].append(r)
        displacement[2].append(s)

        for j in range(3):
            rsens[j].append(receiverstate[j])
            ssens[j].append(senderstate[j])

    ax[0][i].plot(displacement[0],displacement[1],label="receiver")
    ax[0][i].plot(displacement[0],displacement[2],label="sender")
    ax[0][i].plot(displacement[0],[goal]*301,label="goal")
    ax[0][i].set_title(f"Goal = {goal}\nReceiver = {sim.receiverPos:.2f}\nFitness = {sim.fitness():.2f}")
    ax[0][i].legend()
    ax[1][i].plot(displacement[0],rsens[0],label="Contact")
    ax[1][i].plot(displacement[0],rsens[1],label="Self Position")
    # ax[1][i].plot(displacement[0],rsens[2],label="Constant Value")
    ax[1][i].set_title(f"Receiver Sensors")
    ax[1][i].legend()
    ax[2][i].plot(displacement[0],ssens[0],label="Contact")
    ax[2][i].plot(displacement[0],ssens[1],label="Self Position")
    ax[2][i].plot(displacement[0],ssens[2],label="Goal Distance")
    ax[2][i].set_title(f"Sender Sensors")
    ax[2][i].legend()
    i+=1
        # print(sim.getAsciiState())




    print("Final conditions:")
    print("   Sender = {0:.4f}".format(sim.senderPos))
    print(" Receiver = {0:.4f}".format(sim.receiverPos))
    print("     Goal = {0:.4f}".format(sim.goal))
    print(f" fitness = {sim.fitness()}")
    print()

plt.show()
