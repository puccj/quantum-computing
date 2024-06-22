#This has been done in https://amethyst-donkey-pfz7y2up.ws.ide.dwavesys.io/ online IDE.
#I just copy-pasted it here to save it on GitHub


from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel  #Contains function we'll use to define the model we're using
import dwave.inspector

objects = [0,1,2,3]      #Just to label the different objects
weights = [1,2,1,3]     #For each of them: weights and values
values = [-1,-7,-9,-2]  # Negative values because: we want to find the maximim of the function, but the optimizer finds the minimum of the function

max_load = 3

#build variable
x = [f'x_{o}' for o in objects]

#initialize BQM
bqm = BinaryQuadraticModel('BINARY')    #important to add 'BINARY' key, because it recognize that the variables are binary
                                        #another option is to put 'SPIN', which uses -1,1 instead of 0,1. But we've shown the equivalence between the 2
#objective function
for o in objects:
    bqm.add_variable(x[o], values[o])

#constraint function
c = [(x[o], weights[o]) for o in objects]
bqm.add_linear_inequality_constraint(   c,
                                        ub = max_load,
                                        lagrange_multiplier = 5,
                                        label = 'max_load')

# as for IBM which have different topology, the same is true for Dwave: you can connect to qbits only if they are close in some way.
# What this command is doing is taking 4 qbits and choosing physically in the pegasus structure and topology.
# DWave have some default way to do the embedding. It's actually very difficult to find the best embedding for your problem
sampler = EmbeddingComposite(DWaveSampler(solver=dict(topology__type='pegasus')))   
sampleset = sampler.sample(bqm, num_reads = 1000)

sample = sampleset.first.sample

tot_value = 0
tot_weight = 0

for o in objects:
    tot_value -= sample[x[o]]*values[o]
    tot_weight += sample[x[o]]*weights[o]

print('total value:\t', tot_value)
print('total weight:\t', tot_weight)

dwave.inspector.show(sampleset)

#You'll have 2 slack variables to, so the problem is represented by an exagon.
# By clicking on Target-QPU you can see the pegasus topology and the connections.
# You can also see that you embedded the code in 8 qubits. 
# In this particular problem, J_ij is different than 0 for every of the nodes.
# In a plane you can't have no self-intersections
# There are 8 qbits beacuse it's like you split a variable in two,
# keeping the 2 very linked, so that if one is zero also the other. But doing
# that you're able to create the embedding without intersecions

# The library also choose a default annealing time, we've seen that in theoty the slower 
# the better, but in a real physical device, a slower time means that it is also mmore prone to errors and noise