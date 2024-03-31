# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:04:23 2024

@author: huynh
"""

import networkx as nx
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from heapq import *


# Class for implementing Dijskra's algorithm
class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = collections.defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance


# Draws graph and prints result to Console
def drawGraph(graph):
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(
        graph, pos, edge_color='black', width=1, linewidths=1,
        node_size= 250, node_color='pink', alpha=1,
        labels={node: node for node in graph.nodes()}
    )
    plt.show()


# Depth First Search
def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


# Breadth-First Search
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


# Shortest Path Algorithm - Dijsktra

def dijkstra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node
  return visited, path


# Robert Prim's Algorithm for producing a Minimum Spanning Tree
def prim( nodes, edges ):
    conn = defaultdict( list )
    for n1,n2,c in edges:
        conn[ n1 ].append( (c, n1, n2) )
        conn[ n2 ].append( (c, n2, n1) )
 
    mst = []
    used = set( [nodes[ 0 ]] )
    usable_edges = conn[ nodes[0] ][:]
    heapify( usable_edges )
 
    while usable_edges:
        cost, n1, n2 = heappop( usable_edges )
        if n2 not in used:
            used.add( n2 )
            mst.append( ( n1, n2, cost ) )
 
            for e in conn[ n2 ]:
                if e[ 2 ] not in used:
                    heappush( usable_edges, e )
    return mst

def main():
    #-----------------------Depth First/ Breadth First Search algorithms-------------------------#
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
              'P': set(['L'])}
    
    print('Nodes of Graph:', graph.nodes())
    print('\nEdges of Graph:', graph.edges())
    
    drawGraph(graph)
    
    #start_point = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
    
    '''for i in range (len(start_point)):
        print ('Start Point: ', start_point[i])
        depthFirst = dfs(graph1, start_point[i])
        print(depthFirst)'''
        
    
    depthFirst = dfs(graph1, 'A')
    print("DFS: ", depthFirst)

    breadthFirst = bfs(graph1, 'A')
    print ("BFS: ", breadthFirst)
    #-----------------------Dijkstra's Algorithm-------------------------#

    print ("\n\n-----------------------Dijkstra's Algorithm-------------------------")    
    
    graph3 = nx.Graph()
    graph3.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
    graph3.add_edge('A', 'B', weight = 22)
    graph3.add_edge('A', 'C', weight = 9)
    graph3.add_edge('A', 'D', weight = 12)
    graph3.add_edge('B', 'E', weight = 34)
    graph3.add_edge('B', 'C', weight = 35)
    graph3.add_edge('B', 'F', weight = 36)
    graph3.add_edge('C', 'F', weight = 42)
    graph3.add_edge('C', 'E', weight = 65)
    graph3.add_edge('C', 'D', weight = 4)
    graph3.add_edge('D', 'E', weight = 33)
    graph3.add_edge('D', 'I', weight = 30)
    graph3.add_edge('E', 'F', weight = 18)
    graph3.add_edge('E', 'G', weight = 23)
    graph3.add_edge('F', 'H', weight = 24)
    graph3.add_edge('F', 'G', weight = 39)
    graph3.add_edge('G', 'H', weight = 25)
    graph3.add_edge('G', 'I', weight = 21)
    graph3.add_edge('H', 'I', weight = 19)
    
    drawGraph(graph3)
        
    dgraph = Graph()
    dgraph.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}
    dgraph.edges = {'A': ['B', 'C', 'D'], 'B': ['A', 'H', 'C', 'F'], 
                    'C': ['A', 'B', 'F', 'E', 'D'], 'D': ['E', 'I','C','A'], 
                    'E': ['F', 'G', 'D', 'C'], 'F': ['H', 'G', 'E', 'C', 'B'], 
                    'G': ['H', 'I','F', 'E'], 'H': ['I', 'G', 'F', 'B'], 'I': ['H', 'D', 'G']}
    
    dgraph.distance = {('A','B'):22, ('B','A'):22, ('A','C'): 9, ('C','A'):9, ('A','D'):12,('D','A'):12,
                       ('B','H'):34, ('H','B'):34, ('B','C'):35, ('C','B'):35,('B','F'):36,('F','B'):36,
                       ('C','F'):42, ('F','C'):42, ('C','E'):65, ('E','C'):65,('C','D'):4, ('D','C'):4,
                       ('D','E'):33, ('E','D'):33, ('D','I'):30, ('I','D'):30, 
                       ('E','F'):18, ('F','E'):18, ('E','G'):23, ('G','E'):23, 
                       ('F','H'):24, ('H','F'):24, ('F','G'):39, ('G','F'):39,
                       ('G','H'):25, ('H','G'):25, ('G','I'):21, ('I','G'):21,
                       ('H','I'):19, ('I','H'):19}
    
    '''start_point = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for i in range (len(start_point)):
        
        node_visited, path = dijkstra(dgraph, start_point[i])
        print('Visited: ', node_visited)
        print('Path :', path)'''

    node_visited, path = dijkstra(dgraph, 'A')
    print('Visited: ', node_visited)
    print('Path :', path)

#--------------------------Minimum Spanning Trees------------------------#

    nodes = list("ABCDEFGHI")
    edges = [('A', 'B', 22), ('A','C', 9), ('A','D', 12), ('B', 'E', 34), ('B','C', 35), ('B','F', 36),
             ('C','F',42), ('C', 'E',65), ('C','D',4), ('D','E', 33), ('D','I',30), ('E','F',18), ('E','G',23),
             ('F','H', 24), ('F','G',39), ('G','H', 25), ('G','I',21), ('H', 'I', 19)]

    print ('\n\nMinimum Spanning Tree:', prim(nodes,edges))    
if __name__ == "__main__":
    main()
