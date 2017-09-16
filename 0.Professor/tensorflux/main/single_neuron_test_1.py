# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx
import matplotlib.pyplot as plt
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import tensorflux.session as tfs
import datasource.simple as simple_data

### Single_Neuron_Network Test - I
n = tfn.Single_Neuron_Net(input_size=1, output_size=1)
n.initialize()

# Create Input and Target PlaceHolder
x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Value_Assignment.value)
n.layering(activator=tfe.Activator.ReLU.value)

session = tfs.Session()
data = simple_data.Simple_Function_Data()
for train_data in data.training_input:
    output = session.run(n.output, {x: train_data})
    print("Train Data: {:>5}, Feed Forward Output: {:>5}".format(str(train_data), str(output)))

nx.draw_networkx(n, with_labels=True)
plt.show(block=True)