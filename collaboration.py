import networkx as nx

from pyarango import *
from epflgraph import *
import numpy as np
from numpy import linalg
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, euclidean

import heapq

gr = ArangoConnector()


# Function that take a concept id and return its name
def concept_id_to_sring(concept_id):
    assert isinstance(concept_id, int)
    q = """FOR n0 in Nodes_N_Concept
        FILTER n0._key == '""" + str(concept_id) + """'
        RETURN n0.NodeData.PageTitle"""
    output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
    
    return output[0]


# Function that take a concept name and return its id as an int
def concept_string_to_id(concept):
    q = """FOR n0 in Nodes_N_Concept
        FILTER n0.NodeData.PageTitle == '""" + concept + """'
        RETURN n0._key"""
    
    output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
    
    return int(output[0])


# Function that return the neighbours of a concept
def get_neighbours(key):
    assert isinstance(key, int)
    q = """FOR n0 in Nodes_N_Concept
        FILTER n0._key == '""" + str(key) + """'
        FOR c IN Edges_N_Concept_N_Concept_T_GraphScore
        FILTER c.EdgeData.NormalisedScore > 0.2
        FILTER n0._id == c._to
        LET cp = DOCUMENT(c._from)
        LET edge_cp = DOCUMENT(c._id)
        
        RETURN [cp, edge_cp]
        """
    
    output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)

    neighbours = []

    for o in output :
        n = int(o[0]['_key'])
        s = o[1]['EdgeData']['NormalisedScore']
        neighbours.append((n,s))
        
    return neighbours

#Traverses heap looking for specific key
def in_node_heap(heap,node_id): 
    for idx, heap_tuple in enumerate(heap): 
        dist, node = heap_tuple
        if node is node_id: 
            return idx
    return -1 

#Removes element from heap and preserves heap invariant
def delete_in_heap(heap,i):
    heap[i] = heap[-1]
    heap.pop()  # list pop and not heap pop
    if i < len(h):
        heapq._siftup(heap, i)
        heapq._siftdown(heap, 0, i)    


#Select Nodes that are at alpha score-distance of root concept        
def create_graph(root_concept, alpha):
    root_key       = concept_string_to_id(root_concept)
    node_heap      = [(0,root_key)]
    visited_nodes  = [root_key]
    retained_nodes = []
    
    heapq.heapify(node_heap)
    current_dist = 0 
    
    while (current_dist < alpha and len(node_heap) is not 0) : 
        
        current_dist, current_node = heapq.heappop(node_heap)
        # print(current_dist)
        if current_dist < alpha : 
            retained_nodes.append(current_node)
        else: 
            return retained_nodes
        
        neighbours = [(neighbour, 1 - normalized_score if normalized_score < 0.8 else 0.2) for neighbour, normalized_score in get_neighbours(current_node)]
        
        for neighbour, edge_distance in neighbours: 
            if neighbour not in visited_nodes: 
                heap_idx = in_node_heap(node_heap,neighbour)
                if heap_idx is not -1: 
                    current_neighbor_distance, _ = node_heap[heap_idx]
                    if current_neighbor_distance > current_dist + edge_distance: 
                        delete_in_heap(node_heap,heap_idx)
                        heapq.heappush(node_heap,(current_dist+edge_distance,neighbour))
                else:
                    heapq.heappush(node_heap,(current_dist+edge_distance,neighbour))
                visited_nodes.append(neighbour)
    
    return retained_nodes

#Create edges between concepts nodes
def get_concept_edges(G):
    for node in G.nodes():
        q = """FOR n0 in Nodes_N_Concept
        FILTER n0._key == '""" + str(node) + """'
        FOR edge IN Edges_N_Concept_N_Concept_T_GraphScore
        FILTER edge.EdgeData.NormalisedScore > 0.2
        FILTER n0._id == edge._to
        RETURN edge"""
        
        output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
        for x in output:
            n1, n2 = x['_key'].split('_')
            w = x['EdgeData']['NormalisedScore']
            
            n1, n2 = int(n1), int(n2)
            
            if n1 in G and n2 in G:
                G.add_edge(n1, n2, weight=w)
                
#Add prof nodes and edges to concepts 
def add_prof(G):
    concepts = [n for n in G.nodes()]
    
    for node in concepts:
        q = """FOR n0 in Nodes_N_Concept
        FILTER n0._key == '""" + str(node) + """'
        FOR edge IN Edges_N_Person_N_Concept_T_Research
        FILTER n0._id == edge._to
        LET edge_pr = DOCUMENT(edge._id)
        RETURN edge_pr"""
        
        output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
        for edge in output:
            p, c = edge['_key'].split('_')
            w = edge['EdgeData']['NeutralScore']
            
            p, c = int(p), int(c)
            
            if w > 0:
                G.add_node(p, label='Person')
                G.add_edge(p, c, weight=w)
                

# Add publications nodes and edges between profs and or concepts                
def add_publication(G):
    persons = [n for n in G.nodes() if G.nodes[n]['label'] == 'Person']
    concepts = [n for n in G.nodes() if G.nodes[n]['label'] == 'Concept']
    
    for node in persons:
        q = """FOR p in Nodes_N_Person
        FILTER p._key == '""" + str(node) + """'
        FOR edge IN Edges_N_Person_N_Publication
        FILTER p._id == edge._from
        LET edge_pr = DOCUMENT(edge._id)
        RETURN edge_pr"""
        
        output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
        for edge in output:
            p, pub = edge['_key'].split('_')
            p, pub = int(p), int(pub)
            
            G.add_node(pub, label='Publication')
            G.add_edge(p, pub, weight=1.0)
            
    for node in concepts:
        q = """FOR c in Nodes_N_Concept
        FILTER c._key == '""" + str(node) + """'
        FOR edge IN Edges_N_Publication_N_Concept_T_AutoNLP
        FILTER c._id == edge._to
        LET edge_pr = DOCUMENT(edge._id)
        RETURN edge_pr"""
        
        output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
        for edge in output:
            pub, c = edge['_key'].split('_')
            pub, c = int(pub), int(c)
            
            w = edge['EdgeData']['Score']
            
            if pub in G:
                G.add_edge(c, pub, weight=w)

#Compute pagerank, central node is for a given professor
def pagerank(G,damping=0.6,c_thresh=0.001,w_importance=0.5,central_node = None, central_importance = 100000000): 

    beta            = np.array([ central_importance if n is central_node else 1 for n in G.nodes()])
    normalized_beta = beta/linalg.norm(beta,ord=1)
    node_to_index   = {n:i for i,n in enumerate(G.nodes())}
    edge_weights    = nx.get_edge_attributes(G,"weight")
    
    ranks           = np.array([1]*G.number_of_nodes())
    
    A               = np.array([[1 if n2 in nbrdict.keys() else 0 for n2 in G.nodes()]
                                      for n1, nbrdict in G.adjacency()])
    weight_matrix   = np.array([[(float(edge_weights[(n1,n2) if (n1,n2) in edge_weights else (n2,n1)]) if n2 in nbrdict.keys() else 0.0) for n2 in G.nodes()]
                                          for n1, nbrdict in G.adjacency()])

    if central_node is not None: 
        A[node_to_index[central_node]] = np.array([0]*G.number_of_nodes())
        weight_matrix[node_to_index[central_node]] = np.array([0]*G.number_of_nodes())

    summed_weights  = np.sum(weight_matrix, axis=1) 
    degrees         = np.sum(A,axis=1)
    
    wpr_matrix      = (w_importance*(weight_matrix) / summed_weights) + (1 - w_importance)*((A) / degrees)
    if central_node is not None: 
        wpr_matrix[:,node_to_index[central_node]] = normalized_beta[node_to_index[central_node]]      
        
    convergence    = False    
    while (not convergence):
        new_ranks = ( damping*wpr_matrix + (1-damping)*normalized_beta ) @ ranks
        convergence = (linalg.norm((new_ranks-ranks)/ranks,ord=1)/(ranks.shape[0]) < c_thresh )
        ranks = new_ranks
    return {n:ranks[node_to_index[n]] for n in node_to_index.keys()}


#Compute shortest path
def dijkstra_nx(G,root_key,path_endpoints):
    node_heap      = [(0,root_key)]
    visited_nodes  = [root_key]
    retained_nodes = {}
    edge_weights   = nx.get_edge_attributes(G,"weight")
    
    heapq.heapify(node_heap)
    current_dist = 0 
    
    while (len(node_heap) != 0) : 
    
        
        current_dist, current_node = heapq.heappop(node_heap)      
        neighbours = [(neighbour, 1 - normalized_score if normalized_score < 0.8 else 0.2) 
                      for neighbour, normalized_score in zip(list({ k:v for k,v in G.adjacency()}[current_node].keys()), 
                      [edge_weights[(current_node,n)] if (current_node,n) in edge_weights else edge_weights[(n,current_node)] 
                                                                for n in { k:v for k,v in G.adjacency()}[current_node].keys()])]
        
        for neighbour, edge_distance in neighbours: 
            if neighbour not in visited_nodes: 
                heap_idx = in_node_heap(node_heap,neighbour)
                if heap_idx != -1: 
                    current_neighbor_distance, _ = node_heap[heap_idx]
                    if current_neighbor_distance > current_dist + edge_distance: 
                        delete_in_heap(node_heap,heap_idx)
                        heapq.heappush(node_heap,(current_dist+edge_distance,neighbour))
                else:
                    heapq.heappush(node_heap,(current_dist+edge_distance,neighbour))
                visited_nodes.append(neighbour)
        retained_nodes[current_node] = current_dist 
        
    return {k:v for k,v in retained_nodes.items() if k in path_endpoints}
                

#Returns all publications keys for a given prof    
def get_publications(person):      
    q = """FOR p in Nodes_N_Person
    FILTER p._key == '""" + str(person) + """'
    FOR edge IN Edges_N_Person_N_Publication
    FILTER p._id == edge._from
    Let e = Document(edge._to)
    RETURN e._key"""

    output = gr.ExecuteQuery(DatabaseName = "Campus_Analytics", AQLQuery= q)
    return [pub for pub in output]
                

#Function that check if 2 prof have collaborate, ie have publication in common
def check_publications(pub_1, person_2): 
    pub_2 = get_publications(person_2)
    
    return bool(len(np.intersect1d(pub_1, pub_2)))

#For a given prof p, return a dictionaries with other prof keys and corresponding (pagerank, shortestpath, collab) where collab is 1 if true, 0 if false
def get_collab(p, prof, pageranks, shortestpath):
    xaxis = []
    yaxis = []
    collab = []
    
    tmp = prof[:]
    tmp.remove(p)
    
    publi = get_publications(p)
    
    for pp in tmp:
        xaxis.append(pageranks[pp])
        yaxis.append(shortestpath[pp])
            
        if check_publications(publi, pp):
            collab.append(1)
        else :
            collab.append(0)
            
    return {k:v for k,v in zip(tmp,zip(xaxis,yaxis,collab))}



#Compute the geometric median of the cluster of prof that have collaborated
def cluster_center(collabs):
    
    X = np.array([ collabs[pr][0:2] for pr in collabs if collabs[pr][2] ])
    y = np.mean(X, 0)
    
    eps=1e-5

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
        


#Return top 5 prof to recommend
def reccommend(collabs): 
    center_x, center_y = cluster_center(collabs)
    return sorted([((center_x-collabs[pr][0])**2 +(center_y-collabs[pr][1])**2,pr) for pr in collabs if collabs[pr][2]==0])[:5]
        
#Plot the collaborators, other profs and recommended collaborators. Save the plot
def plot_p(p, collabs):
    collab_x = []
    collab_y = []
    nop_x    = []
    nop_y    = []
    rec_x    = []
    rec_y    = []
    
    top_5 = [node for _ , node in reccommend(collabs)]
    
    for pr in collabs:
        if collabs[pr][2] :
            collab_x.append(collabs[pr][0])
            collab_y.append(collabs[pr][1])
        elif pr in top_5: 
            rec_x.append(collabs[pr][0])
            rec_y.append(collabs[pr][1])
        else:
            nop_x.append(collabs[pr][0])
            nop_y.append(collabs[pr][1])
    
    rgba_colors = np.zeros((len(nop_x+collab_x+rec_x),4))
    rgba_colors[:len(nop_x)] = [1,0,0,0.1]
    rgba_colors[len(nop_x):-len(rec_x)] = [0,0,1,1]
    rgba_colors[-len(rec_x):] = [0,1,0,1]
    
    plt.title('Plot for Researcher id.' + str(p))
    plt.xlabel('PageRank Score')
    plt.ylabel('Shortest Path Distance')
    s1 = plt.scatter(nop_x,nop_y, s=10, c=rgba_colors[:len(nop_x)])
    s2 = plt.scatter(collab_x,collab_y, s=10, c=rgba_colors[len(nop_x):-len(rec_x)])
    s3 = plt.scatter(rec_x,rec_y, s=10, c=rgba_colors[-len(rec_x):])
    plt.legend((s1,s2,s3),('Other Rearchers','Past Collaborators','Reccommended Collaborators'))
    plt.savefig(str(p)+'.png')
    plt.show()