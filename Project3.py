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
import time
import heapq
#import resource
from itertools import groupby


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
    
# Tracker to help keep track of various components
class Tracker(object):
    """Keeps track of the current time, current source, component leader,
    finish time of each node and the explored nodes.
    
    'self.leader' is informs of {node: leader, ...}."""

    def __init__(self):
        self.current_time = 0
        self.current_source = None
        self.leader = {}
        self.finish_time = {}
        self.explored = set()


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


# Depth First Search variant with Tracker; for Question 2
def dfs_tracker(graph_dict, node, tracker):
    """Inner loop explores all nodes in a SCC. Graph represented as a dict,
    {tail: [head_list], ...}. Depth first search runs recursively and keeps
    track of the parameters"""

    tracker.explored.add(node)
    tracker.leader[node] = tracker.current_source
    for head in graph_dict[node]:
        if head not in tracker.explored:
            dfs_tracker(graph_dict, head, tracker)
    tracker.current_time += 1
    tracker.finish_time[node] = tracker.current_time

# Depth First Search Loop; for Question 2
def dfs_loop(graph_dict, nodes, tracker):
    """Outer loop checks out all SCCs. Current source node changes when one
    SCC inner loop finishes."""

    for node in nodes:
        if node not in tracker.explored:
            tracker.current_source = node
            dfs_tracker(graph_dict, node, tracker)


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


# Graph Reverse; for Question 2
def graph_reverse(graph):
    """Given a directed graph in forms of {tail:[head_list], ...}, compute
    a reversed directed graph, in which every edge changes direction."""

    reversed_graph = defaultdict(list)
    for tail, head_list in graph.items():
        for head in head_list:
            reversed_graph[head].append(tail)
    return reversed_graph


# SCC; for Question 2
def scc(graph):
    """First runs dfs_loop on reversed graph with nodes in decreasing order,
    then runs dfs_loop on original graph with nodes in decreasing finish
    time order(obtained from first run). Return a dict of {leader: SCC}."""

    out = defaultdict(list)
    tracker1 = Tracker()
    tracker2 = Tracker()
    nodes = set()
    reversed_graph = graph_reverse(graph)
    for tail, head_list in graph.items():
        nodes |= set(head_list)
        nodes.add(tail)
    nodes = sorted(list(nodes), reverse=True)
    dfs_loop(reversed_graph, nodes, tracker1)
    sorted_nodes = sorted(tracker1.finish_time,
                          key=tracker1.finish_time.get, reverse=True)
    dfs_loop(graph, sorted_nodes, tracker2)
    for lead, vertex in groupby(sorted(tracker2.leader, key=tracker2.leader.get),
                                key=tracker2.leader.get):
        out[lead] = list(vertex)
    return out


# Linearizes a graph in its topological order; for Question 2
def dfs_tpl_order(graph,start,path,n):
    path = path + [start]
    for edge in graph[start]: 
        if edge not in path:
            path = dfs_tpl_order(graph, edge,path,n)
    print (n, start)
    n -= 1
    return path

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
    
    start_point = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
    start_bool = [False, False, False, False, False, False, False, False,False, False, False, False,False, False, False, False]
    
    depthFirst = dfs(graph1, 'A')
    
    #print (list(depthFirst)[0])
    
    for i in range (len(list(depthFirst))):
        j = 0
        for j in range (len(start_point)):
            
            if start_point[j] == list(depthFirst)[i]: 
                start_bool[j] = True
                break
            
    for i in range (len(start_bool)):
        if start_bool[i] == False:
            depthDC = dfs(graph1, start_point[i])
            break
        
    print (start_bool)
    print("DFS: ", depthFirst, depthDC)
    
#------------------------------------Breadth First----------------------------------------------#
    
    start_bool = [False, False, False, False, False, False, False, False,False, False, False, False,False, False, False, False]
    breadthFirst = bfs(graph1, 'A')
    
    for i in range (len(list(breadthFirst))):
        j = 0
        for j in range (len(start_point)):
            
            if start_point[j] == list(breadthFirst)[i]: 
                start_bool[j] = True
                break
            
    for i in range (len(start_bool)):
        if start_bool[i] == False:
            breadthDC = bfs(graph1, start_point[i])
            break
    
   
    print ("BFS: ", breadthFirst, breadthDC)
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


    #----------------------Question 2 - Hunter Henderson--------------------#
    start = time.time()
    '''    graph = defaultdict(list)
    with open('SCC.txt') as file_in:
    #with open('test.txt') as file_in:
        for line in file_in:
            x = line.strip().split()
            x1, x2 = int(x[0]), int(x[1])
            graph[x1].append(x2)'''
            
    dgraph2 = nx.Graph()
    dgraph2.add_nodes_from(['1', '2','3','4','5','6','7','8','9','10','11','12'])
    
    dgraph2.add_edges_from([('1', '3')])
    dgraph2.add_edges_from([('2', '1')])
    dgraph2.add_edges_from([('3','2'), ('3', '5')])
    dgraph2.add_edges_from([('4','4'),('4','2'), ('4', '12')])
    dgraph2.add_edges_from([('5', '6'), ('5', '8')])
    dgraph2.add_edges_from([('6', '7'), ('6', '8'), ('6', '10')])
    dgraph2.add_edges_from([('7', '10')])
    dgraph2.add_edges_from([('8', '9'), ('8', '10')])
    dgraph2.add_edges_from([('9', '5'), ('9', '11')])
    dgraph2.add_edges_from([('10', '9'), ('10', '11')])
    dgraph2.add_edges_from([('11', '12')])
    
    
    graph2 = {'1': set(['3']),
             '2': set(['1']),
             '3': set(['2', '5']),
             '4': set(['1', '2', '12']),
             '5': set(['6', '8']),
             '6': set(['7', '8', '10']),
             '7': set(['10']),
             '8': set(['9', '10']),
             '9': set(['5', '11']),
             '10': set(['9', '11']),
             '11': set(['12']),
             '12': set()}
    t1 = time.time() - start
    print (t1)
    groups = scc(graph2)
    t2 = time.time() - start
    print (round(t2,4))
    top_5 = heapq.nlargest(5, groups, key=lambda x: len(groups[x]))
    #sorted_groups = sorted(groups, key=lambda x: len(groups[x]), reverse=True)
    result = []
    for i in range(5):
        try:
            result.append(len(groups[top_5[i]]))
            #result.append(len(groups[sorted_groups[i]]))
        except:
            result.append(0)
            
    print('\n\nQuestion 2A: Strongly connected components are:')
    for key in groups:
        print(groups[key])
        
    # print('\n\nQuestion 2B: Meta Graph Diagraph')
    # drawGraph(dgraph2)
    
    n = len(graph2)
    print('\n\nQuestion 2C: Topological order starting from \'1\'')
    u = dfs_tpl_order(graph2, '1', [], n)
    print(u)
    
#----------------------End of Program--------------------#

if __name__ == "__main__":
    main()