__author__ = 'babak_khorrami'

import numpy as np
from math import tanh

# Build the network:
def network(n):
    edge=[]
    for i in range(0,n):
        for j in range(1,n):
            edge.append(tuple((n*i+j,n*i+j+1)))

    for i in range(0,n-1):
        for j in range(1,8):
            edge.append(tuple((n*i+j,n*(i+1)+j)))

    for i in range(1,n+1):
        edge.append(tuple((i,(n-1)*n+i)))

    for i in range(0,n):
        edge.append(tuple((n*i+1,n*(i+1))))

    ed1 = np.array(edge)
    ed2 = np.c_[ed1[:,1],ed1[:,0]]
    ed = np.r_[ed1,ed2]

    return ed

# Helper function to find the neighbors of a given node in the network
def get_neighbors(edges,node):
    ind = np.where(edges[:,0]==node)
    return edges[ind,1]

# Gibbs Sampler:
def gibbs_sampler(edges,node_count,node_theta,edge_theta,burn_in,sample_size):
    #initialize the grid nodes to +1 , -1
    config=np.random.randint(0,2,node_count).reshape((node_count))
    ind=np.where(config==0)
    config[ind]=-1
    samples = np.zeros((sample_size+burn_in,node_count)) #place-holder for samples
    iteration = burn_in + sample_size #total # iterations
    samples=[]
    for i in range(1,iteration+1):
        new_config = np.zeros(node_count)
        for n in range(1,node_count+1): #Nodes are ordered 1,...,49
            neighb = get_neighbors(edges,n)
            idx = np.array(neighb-1)
            exponent = node_theta**n + edge_theta*np.sum(config[idx])
            x=config[n-1]
            if x==1:
                prob = np.exp(exponent)/(np.exp(exponent)+np.exp(-exponent))
            elif x==-1:
                prob = np.exp(-exponent)/(np.exp(exponent)+np.exp(-exponent))
            else:
                break
            u=np.random.random(1) #uniform [0,1] RN
            if u<prob:
                new_config[n-1] = -1
            else:
                new_config[n-1] = +1
        if i>=burn_in:
            samples.append(new_config)

    return samples

#Naive Mean Field for Ising Grid :
def naive_mean_field(edges,node_count,node_theta,edge_theta):
    #initialize the grid nodes to +1 , -1
    config=np.random.uniform(-1,1,node_count).reshape((node_count))
    t=1e-6 #threshold
    count = 0 #iterations
    # old_config = np.zeros(node_count)
    change = 1
    while change > t:
        old_config=config.copy()
        for n in range(1,node_count+1): #Nodes are ordered 1,...,49
            neighb = get_neighbors(edges,n)
            idx = np.array(neighb-1)
            exponent = node_theta**n + edge_theta*np.sum(config[idx])
            num = tanh(exponent)
            config[n-1] = num
        change = np.linalg.norm(config - old_config,np.inf)
        count = count + 1

    return config , count


edges = network(7) #construct the grid
print(edges)
node_count = 49
node_theta = 0.25
edge_theta = -1
burn_in = 1000
sample_size = 15000
smp = gibbs_sampler(edges,node_count,node_theta,edge_theta,burn_in,sample_size)
sample = np.array(smp)

means = np.mean(sample,axis=0)
print(means)
# id=np.array(range(1,50))
# results=np.c_[id,means]
# print(results)
# x=np.round(means.reshape((7,7)),4)
# print(x)
# np.savetxt("foo.csv", x, delimiter=",")

mean_field, count = naive_mean_field(edges,node_count,node_theta,edge_theta)
print("---------------------------------------------------------")
print(mean_field , count)
print("Difference : ",np.mean(np.abs(means-mean_field)))









