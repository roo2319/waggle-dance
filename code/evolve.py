import ctrnn
import statistics
import random

mutationRate = 0.447
centerCrossing = False
class Citizen():
    def __init__(self,genome=None,fitness=0,age=0):
        if genome == None:
            genome = ctrnn.Genome(centerCrossing=centerCrossing)
        self.genome = genome
        self.fitness = fitness
        self.age = age
        # Fitness for the purpose of sus
        self.rfitness = None
    
    def copy(self):
        return Citizen(self.genome.copy(),self.fitness,self.age)

def assess(pop, pool, fitness,rs):
    """
    Calculate fitness score for each member across the given population

    Parameters:
        pop (list) : A list of members of the population
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        fitness (func) : A function with type Genome, Task -> Float

    Return:
        The population, sorted by their newly assessed fitness scores 
    """

    pop = [(x,fitness,rs) for x in pop]
    pop = pool.starmap(assess_item, pop)
    return sorted(pop, key = lambda i: i.fitness, reverse=True) 

def assess_item(item,fitness,rs):
    """
    Calculate the fitness of a single genome

    Parameters:
        item : A single member of the population
        fitness (func) : A function with type Genome, Task -> Float
    
    Return:
        item, with an updated fitness value

    """
    item.fitness = fitness(item.genome,rs)
    return item

def mutate(pop, pool,fitness,rs):
    """
    Mutate each member of the population

    Parameters:
        pop (list) : A list of members of the population
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        fitness (func) : A function with type Genome, Task -> Float

    Return:
        pop, with a mutation applied to each member

    """
    pop = [(x,fitness,rs) for x in pop]
    result = pool.starmap(mutate_item,pop)
    pop = [x[0] for x in result]
    
    # pop = [x[0] for x in pop]
    return sorted(pop, key = lambda i: i.fitness, reverse=True), sum(x[1] for x in result)

def mutate_item(item,fitness,rs):
    """
    Mutate a single genome, only keep the result if it's better than the parent

    Parameters:
        item : A single member of the population
        fitness (func) : A function with type Genome, Task -> Float
    
    Return:
        item, with a mutated genome (if it has better fitness than the original)
    """

    child = item.genome.copy()
    child.beerMutate(mutationRate)
    cfitness = fitness(child,rs)
    if cfitness >= item.fitness:
        return Citizen(child,cfitness), 1
    item.age += 1
    return item, 0

def rank_reduce(fitnesses):
    """
    Calculate a weighted sum of fitnesses, with stronger weighting to lower scores

    Parameters:
        fitnesses (list(float))
    
    Return:
        A single float, representing overall fitness
    """

    s_fitnesses = sorted(fitnesses)
    return sum([elem/(i+1) for i,elem in enumerate(s_fitnesses)])

def min_fitness(fitnesses):
    return min(fitnesses)

def initialise(pop_size):
    """
    Initialise a population of random genomes

    Parameters:
        pop_size (int) : The number of members of the seed population
        pop_size (int) : The number of members population members to be returned
        pool (multiprocessing.pool.Pool) : A multiprocessing pool
        fitness (func) : A function with type Genome, Task -> Float
        
    Return:
        A list containing pop_size members
    """

    pop = []
    while len(pop)<pop_size:
        pop.append(Citizen())

    return pop

def selection(pop,elitism=0,size=None):
    elites = pop[:elitism]
    sel = rank_roulette_select(pop[elitism:],size)
    return sorted(elites+sel, key = lambda i: i.fitness, reverse=True) 


def rank_roulette_select(pop,size=None):
    """
    Perform roulette selection, weighted by rank rather than fitness

    Parameters:
        pop : A sorted list of population members

    Return:
        A new population chosen by rank based roulette selection
    """
    if size == None:
        size = len(pop)
    pop = random.choices(pop,reversed(range(1,len(pop)+1)),k=size)
    return pop

def sus(pop):
    # Bakers stochastic universal sampling
    MaxExpOffspring = 1.1
    size = len(pop)
    # Rerank using bakers linear ranking method
    for count, i in enumerate(pop):
        i.rfitness = (MaxExpOffspring + (2 - 2*MaxExpOffspring)*((count)/(size-1)))/(size)

    rand = random.random() 
    total = 0
    i = -1
    newpop = []
    while len(newpop) < size:
        if rand < total:
            # if you don't create a new citizen bad things happen
            newpop.append(Citizen(pop[i].genome,pop[i].fitness,pop[i].age))
            rand += 1
            continue
        i += 1
        total += size * pop[i].rfitness
    return newpop

    

def truncation_select(pop):
    """
    Replace 0 fitness members of the population with strong fitness members

    Parameters:
        pop (list) : The initial population

    Return:
        The updated population
    """

    size = len(pop)
    survivors = (9*size)//10
    pop = pop[:survivors]
    for i in range(size-survivors):
        pop.append(pop[i].copy())
    return pop 



def log_fitness(pop, gen, mcount, file=None):
    """
    Write aggregate statistics of the population, either to a file or to stdout

    Parameters:
        pop (list) : The population
        gen (int) : The generation number
        file (file) : The file that will be written to, None -> Stdout
    """
    
    fitness = [x.fitness for x in pop]
    ages = [x.age for x in pop]
    fline = "{:4d} - Fitness  : max:{:.3f}, min:{:.3f}, mean:{:.3f}".format(gen,max(fitness),min(fitness),statistics.mean(fitness))
    aline = "{:4d} - Age      : max:{:.3f}, min:{:.3f}, median:{:.3f}".format(gen,max(ages),min(ages),statistics.median(ages))
    mline = f"{gen:4d} - Mutations: {mcount}"

    if file:
        file.write(fline+"\n"+aline+"\n"+mline+"\n")
    else:
        print(fline+"\n"+aline+"\n"+mline+"\n")

def main():
    pass

if __name__ == '__main__':
    main()
    