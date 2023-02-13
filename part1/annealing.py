import networkx as nx
from pyqubo import Binary
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
import matplotlib.pyplot as plt

########################
####   入力データ    #####
########################
# アニーリング設定
weight = 1000
SHOTS = 1000
DWAVE_TOKEN=""

########################
###   グラフデータ    ####
########################
G = nx.DiGraph()
G.add_edges_from([('s', 1), (1, 2), (1, 'g')])
nx.draw(G, node_color="#414175", font_color="#fff", with_labels=True)
plt.show()

########################
####    QUBO変換    #####
########################
r_s1 = Binary("s->1")
r_12 = Binary("1->2")
r_1g = Binary("1->g")

QUBO_Obj = r_s1 + r_12 + r_1g
QUBO_Penalty = (r_s1 - 1) ** 2 + (r_s1 - r_12 - r_1g) ** 2 + (1 - r_1g) ** 2
QUBO = QUBO_Obj + weight * QUBO_Penalty

########################
##   量子アニーリング    ##
########################
qubo_model = QUBO.compile()
qubo_compiled, offset = qubo_model.to_qubo()

if not DWAVE_TOKEN:
        sampleset = SimulatedAnnealingSampler().sample_qubo(qubo_compiled, num_reads=SHOTS)
else:
    sampler = EmbeddingComposite(DWaveSampler(token=DWAVE_TOKEN))
    sampleset = sampler.sample_qubo(qubo_compiled, num_reads=SHOTS)

answer = sampleset.first.sample
energy = sampleset.first.energy
print(answer)
print(f"Energy: {energy}")