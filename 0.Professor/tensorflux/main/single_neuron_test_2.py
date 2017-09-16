# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx
import matplotlib.pyplot as plt
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import tensorflux.session as tfs
import datasource.simple as simple_data
import numpy as np

### Single_Neuron_Network Test - II
n = tfn.Single_Neuron_Net(input_size=2, output_size=1)
n.initialize()

# Create Input and Target PlaceHolder
x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Truncated_Normal.value)
n.layering(activator=tfe.Activator.ReLU.value)

session = tfs.Session()
data = simple_data.Or_Gate_Data()
for train_data in data.training_input:
    output = session.run(n.output, {x: train_data})
    print("Train Data: {:>5}, Feed Forward Output: {:>20}".format(str(train_data), np.array2string(output)))
    print()

nx.draw_networkx(n, with_labels=True)
plt.show(block=True)
