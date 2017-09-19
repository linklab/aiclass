# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs
import numpy as np
from tensorflux import graph as tfg


class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):#Session은 operation만 받는다.
        """Computes the output of an operation
        
        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = self.traverse_postorder(operation) #리스트

        for node in nodes_postorder:
            print(node)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:
            if type(node) == tfg.Placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]# key == node
            elif type(node) == tfg.Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value #output변수를 새로 만듦.
            else: # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]
                # operantion(+, *,..)에 [5.0, 1.0,..]등의 placeholder나 variable이 리스트로 들어감

                # Compute the output of this operation
                node.output = node.forward(*node.inputs) #*node.inputs리스트 내의 원소들을 의미
                #5.0*1.0

            # Convert lists to numpy arrays
            if type(node.output) == list: #5.0 등의 스칼라값이 아닌 리스트라면,
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output

    @staticmethod #이 라인은 해도 되고 안해도 되는 부분
    def traverse_postorder(operation):
        """Performs a post-order traversal, returning a list of nodes
        in the order in which they have to be computed

        Args:
           operation: The operation to start traversal at
        """

        nodes_postorder = [] #inorder, preorder, postorder 부모노드가 앞이냐 가운데냐 뒤냐
        # 연산의 순서가 postorder가 맞다.

        def recursive_visit(node):
            if isinstance(node, tfg.Operation):# operation객체의 인스턴스라면
                for input_node in node.input_nodes:
                    recursive_visit(input_node)
            nodes_postorder.append(node)

        recursive_visit(operation)
        return nodes_postorder