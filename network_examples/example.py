# load a network
from epynet import Network
network = Network('large.inp')
# solve network
network.solve()
# properties
#print(network.pipes['4'].flow)
#print(network.nodes['1'].demand)
# valve manipulation
#network.valves['12'].setting = 10
# convinience properties
#print(network.pipes['5'].downstream_node.pressure)
#print(network.nodes['1'].upstream_links[0].velocity)
# pandas integration
#print(network.pipes.flow)
#print(network.pipes.length[network.pipes.velocity > 1])
#print(network.nodes.demand[network.nodes.pressure < 10].max())
# network manipulaton
network.add_tank('tankid', x=10, y=10, tanklevel=100, diameter=200,minlevel=0,maxlevel=500)
network.add_junction('junctionid', x=20, y=10, elevation=5)
network.add_pipe(uid="pipeid",length=10, diameter=200, roughness=0.1, from_node="tankid",to_node="junctionid")