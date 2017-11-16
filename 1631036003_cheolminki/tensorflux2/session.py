# -*- coding: utf-8 -*-
# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs
import numpy as np
from tensorflux import graph as tfg


class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):     #operation을 받기 때문에 variable 사용 불
        """Computes the output of an operation
        
        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = self.traverse_postorder(operation)

        for node in nodes_postorder:
            print(node)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:
            if type(node) == tfg.Placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == tfg.Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else: # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.forward(*node.inputs)    #node.inputs는 리스트, *가 붙으면 값을 불러와서 집어넣어줌

            # Convert lists to numpy arrays
            if type(node.output) == list:          #list type이면 numpy array로 바꿈
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output

    @staticmethod
    def traverse_postorder(operation):      #class의 메소드
        """Performs a post-order traversal, returning a list of nodes
        in the order in which they have to be computed

        Args:
           operation: The operation to start traversal at
        """

        #post-order방식으로 그래프의 순서를 정함.
        nodes_postorder = []


        def recursive_visit(node):  #method 내의 method 선언
            if isinstance(node, tfg.Operation):
                for input_node in node.input_nodes:
                    recursive_visit(input_node)
            nodes_postorder.append(node)

        recursive_visit(operation)
        return nodes_postorder