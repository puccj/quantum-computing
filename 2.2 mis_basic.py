#This has been done in https://amethyst-donkey-pfz7y2up.ws.ide.dwavesys.io/ online IDE.
#I just copy-pasted it here to save it on GitHub


from dwave.system import DWaveSampler, EmbeddingComposite   #in this case simpler than the other
from collections import defaultdict
import dwave.inspector

import networkx as nx

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
G.add_edges_from([(1,4), (1,5), (1,7), (2,4), (2,7), (3,7), (3,5), (4,5), (4,7), (5,7), (6,7)])

nx.draw(G)

# Initialize our h vector, J matrix
h = defaultdict(int)
J = defaultdict(int)

# Update J matrix for every edge in the graph
# n = (1-z)/2   ->  replace n with this, expand and we find:
for i,j in G.edges:
    J[(i,j)] += 2
    h[i] -= 1
    h[j] -= 1

# ------ Run our QUBO on the QPU -------
# Set up the QPU parameters
numruns = 100

sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising(h, J, num_reads=numruns)

print('solution:\t', response.first)
dwave.inspector.show(response)

# You can see that it uses 8 qbits. In my case I had 2 variable '7', prof had 2 variable '5'
#