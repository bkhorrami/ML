__author__ = 'babak_khorrami'

import numpy as np
import pandas as pd
from collections import *
from itertools import *

class Tree(object):
    def __init__(self,root = 1 ,edges = Counter(), nodes = Counter(), messages = Counter()):
        self.root = root
        self.edges = edges  #Edges of the tree / Dictionary
        self.nodes = nodes  #Nodes of the tree /Dictionary
        self.message = messages #Messages being sent (An Array from s to t for (s,t) in E)
        tmp_edge=[]
        for e in self.edges.keys():
            tmp_edge.append(e)
        edge_list=np.array(tmp_edge) #List of edges
        self.edge_list = edge_list
        self.node_list=list(self.nodes.keys())
        self.marginals = np.zeros((len(self.node_list),2))
        self.marginals_dict=Counter(self.node_list)


    def get_cmp_fnc_node(self,n):
        return self.nodes[n]


    def get_cmp_fnc_edge(self,e):
        return self.edges[e]


    def get_all_messages(self):
        return self.message


    def get_message(self,e):
        """
        get message of edge : e
        """
        return self.message[e]


    def find_leaf(self):
        """
        Find the leaves of the tree at each iteration
        :return: leaf nodes
        """
        ee=self.edges.keys()
        heads=[x[0] for x in ee]
        tails=[x[1] for x in ee]
        nd=Counter(list(chain(heads,tails)))
        leaf=[]
        for k,v in nd.items():
            if v==2:
               leaf.append(k)

        return leaf

    def find_neighbors(self,node):
        """
        :param node:
        :return: Returns the neighbors of the given node
        """
        ind=np.in1d(self.edge_list[:,0],node)
        neighbors = self.edge_list[ind,1]
        return neighbors

    def initialize_messages(self,compfunc_edges):
        self.message=Counter(self.edges.keys())
        cfe=np.array(compfunc_edges)
        for k in self.message.keys():
            self.message[k]=np.ones((1,cfe.shape[1]))


    # def send_message(self,e):
    #     """*** Send message from node n1 to n2 e=(n1,n2):
    #     """
    #     n1 , n2 = e[0] , e[1]
    #     phi_n1 =  self.get_cmp_fnc_node(n1)
    #     all_neigh_n1=self.find_neighbors(n1)
    #     neigh_n1 = np.setdiff1d(all_neigh_n1,n2)
    #     for u in neigh_n1:
    #         phi_n1 = np.dot(phi_n1,self.get_cmp_fnc_edge(tuple((u,n1))))
    #
    #     phi_n1n2 = self.get_cmp_fnc_edge(tuple((n1,n2)))
    #     dup_phi_n1 = np.tile(phi_n1,(2,1))
    #     self.message[e] = np.sum(self.message[e]*dup_phi_n1,axis=0)


    def send_message(self,e):
        """
        Send message from node n1 to n2, e=(n1,n2):
        """
        n1 , n2 = e[0] , e[1]
        phi_n1 =  self.get_cmp_fnc_node(n1)
        all_neigh_n1=self.find_neighbors(n1)
        neigh_n1 = np.setdiff1d(all_neigh_n1,n2)
        phi_n1n2 = self.get_cmp_fnc_edge(tuple((n1,n2)))
        for u in neigh_n1:
            phi_n1 = np.dot(phi_n1 , self.get_message(tuple((u,n1))))

        dup_phi_n1 = np.tile(phi_n1,(2,1))
        self.message[e] = np.sum(self.message[e]*dup_phi_n1,axis=0)


    def collect_message(self,e):
        n1 , n2 = e[0] , e[1]
        all_neigh_n2=self.find_neighbors(n2)
        neigh_n2 = np.setdiff1d(all_neigh_n2,n1)
        for u in neigh_n2:
            self.collect_message(tuple((n2,u)))
        self.send_message(tuple((n2,n1)))


    def distribute_message(self,e):
        n1 , n2 = e[0] , e[1]
        self.send_message(tuple((n1,n2)))
        all_neigh_n2=self.find_neighbors(n2)
        neigh_n2 = np.setdiff1d(all_neigh_n2,n1)
        for u in neigh_n2:
            self.distribute_message(tuple((n2,u)))


    def collect_to_root(self):
        root_neigh = self.find_neighbors(self.root)
        for n in root_neigh:
            self.collect_message(tuple((self.root,n)))


    def distribute_from_root(self):
        root_neigh = self.find_neighbors(self.root)
        for n in root_neigh:
            self.distribute_message(tuple((self.root,n)))


    def calculate_marginals(self):
        count=0
        for n in self.node_list:
            phi_n1 =  self.get_cmp_fnc_node(n)
            n_neigh = self.find_neighbors(n)
            for u in n_neigh:
                phi_n1 = phi_n1 * self.get_message(tuple((u,n)))
            self.marginals[count,:] = phi_n1/np.sum(phi_n1)
            count+=1
            self.marginals_dict[n] = phi_n1/np.sum(phi_n1)

    def get_marginals(self):
        return self.marginals

    def get_marg_dict(self):
        return self.marginals_dict


#************ Build the Tree **********
#*** Edges:
ce=[[1,0.45],[0.45,1]] #compatibilty function for edges
cn135=[0.7,0.3] #compatibilty function for nodes 1,3,5
cn246=[0.1,0.9] #compatibilty function for nodes 2,4,6

eds = pd.read_csv("/Users/babak_khorrami/Documents/pyCode/sum_product/edges.csv")
ed=np.array(eds)
tp=list(zip(ed[:,0],ed[:,1]))
tree_edges=Counter(tp)
for k in tree_edges.keys():
    tree_edges[k]=np.array(ce)

#*** Nodes:
tn=list(range(1,7))
tree_nodes=Counter(tn)
for k in tree_nodes.keys():
    if k%2==0:
        tree_nodes[k]=np.array(cn246)
    else:
        tree_nodes[k]=np.array(cn135)

#*** Initialize Messages:
message=Counter(tp)
# for k in message.keys():
#     message[k]=np.array(np.ones(1,(np.array(ce)).shape[1]))

tree_42a=Tree(1,tree_edges,tree_nodes,message)
tree_42a.initialize_messages(ce)
tree_42a.collect_to_root()
tree_42a.distribute_from_root()
tree_42a.calculate_marginals()
print(tree_42a.get_marginals())
print("******--------------------------******")
dict=tree_42a.get_marg_dict()
for i in dict.items():
    print(i)

print(tree_42a.get_message(tuple((2,4))))
print(np.linalg.norm(tree_42a.get_message(tuple((2,4)))-tree_42a.get_message(tuple((1,2))),np.inf))


# #------ TEST :
# tree_test=Tree(1,tree_edges,tree_nodes,message)
# tree_test.initialize_messages(ce)
# old_message = tree_test.get_all_messages() #get the intial message as a starting point
# max_iter = 1000
# om=list(old_message.values())
# tree_test.collect_to_root()
# tree_test.distribute_from_root()
# new_message = tree_test.get_all_messages()
# nm=list(new_message.values())
# dist = []
# for i in range(len(nm)):
#     dist.append(np.linalg.norm(om[i]-nm[i],np.inf))
#
# print(dist)

# for t in range(max_iter):
#     tree_test.collect_to_root()
#     tree_test.distribute_from_root()
#     new_message = tree_test.get_all_messages()







# train=pd.read_csv("/Users/babak_khorrami/Downloads/train 2.csv")
# print(train.head(1))'
# print(np.unique(train['cont5']))

# cc=pd.read_csv("https://people.eecs.berkeley.edu/~bartlett/courses/2009fall-cs281a/hw5-1.true")
# print(cc.head())
