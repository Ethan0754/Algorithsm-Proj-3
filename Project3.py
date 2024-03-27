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

def _dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
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
graph.add_edges_from([('H',' K'), ('H', 'L'), ('K','L'),('K','O'),('L', 'P')])

graph1 = {'A': set(['B', 'E', 'F']),
          'B': set(['C', 'F']),
          'C': set(['D','G']),
          'D': set(['G']),
          'E': set(['F', 'I']),
          'F': set(['I']),
          'G': set(['D']),
          'H': set(['K', 'L']),
          'I': set(['J', 'M']),
          'J': set(['G']),
          'K': set(['L', 'O']),
          'L': set(['P']),
          'M': set(['N']),
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
    
depthFirst = _dfs(graph1, 'A')
print(depthFirst)

breadthFirst = bfs(graph1, 'A')
print (breadthFirst)

