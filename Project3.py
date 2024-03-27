# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:04:23 2024

@author: huynh
"""

import networkx as nx
import matplotlib.pyplot as plt


def drawGraph(graph):
    
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(
        graph, pos, edge_color='black', width=1, linewidths=1,
        node_size= 250, node_color='pink', alpha=1,
        labels={node: node for node in graph.nodes()}
    )
    plt.show()

def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        #print('The stack is:', stack)
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def bfs(graph, start):
    visited, queue = set(), [start]
    p =[]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            p.append(vertex)
            queue.extend(graph[vertex] - visited)
    return p


graph = nx.Graph()
graph.add_nodes_from(['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'])

graph.add_edges_from([('A', 'B'), ('A', 'E'), ('A', 'F')])
graph.add_edges_from([('B', 'C'), ('B', 'F')])
graph.add_edges_from([('C','D'), ('C', 'G'), ('D','G')])
graph.add_edges_from([('E','F'),('E','I'), ('F', 'I')])
graph.add_edges_from([('I', 'J'), ('I', 'M'), ('M','N')])
graph.add_edges_from([('J', 'G')])
graph.add_edges_from([('H', 'K'), ('H','L')])
graph.add_edges_from([('K', 'O'), ('K', 'L')])
graph.add_edges_from([('L', 'P')])

graph1 = {'A': set(['B', 'E', 'F']),
          'B': set(['A' ,'C', 'F']),
          'C': set(['B', 'D','G']),
          'D': set(['G']),
          'E': set(['A', 'F', 'I']),
          'F': set(['E', 'I']),
          'G': set(['C', 'D']),
          'H': set(['K', 'L']),
          'I': set(['E' ,'J', 'M']),
          'J': set(['I', 'G']),
          'K': set(['H', 'L', 'O']),
          'L': set(['H','K','P']),
          'M': set(['I', 'N']),
          'N': set(['M']),
          'O': set(['K']),
          'P': set(['L'])

         }

print('Nodes of Graph:', graph.nodes())
print('\nEdges of Graph:', graph.edges())

drawGraph(graph)

start_point = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']

'''for i in range (len(start_point)):
    print('Start Point: ', start_point[i])
    z = _dfs(graph1, start_point[i])
    print(z)'''

depthFirst = dfs(graph1, 'H')
print("DFS: ", depthFirst)

breadthFirst = bfs(graph1, 'H')
print ("BFS: ", breadthFirst)

