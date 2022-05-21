# Redes de Computadores 2022
# Universidad del Rosario - School of Engineering, Science and Technology
# Pr. David Celeita
from re import A
import numpy as np
import pandas as pd
import copy
import scipy as sp
import networkx as nx
from scipy import sparse
import scipy.optimize as opt
import time
import random as rd
import csv
import matplotlib.pyplot as plt

class Node:
    # Class that defines a node of a graph
    def __init__(self, id):
        # Class constructor
        # INPUT:
        #   id: int that identifies node
        self.id = id
        self.neighs = []
        self.distance = np.inf
        self.predecessor = None

    def __str__(self):
        ret = str(self.id) + " neighs "
        for i in self.neighs:
            ret += str(i[0].id) + ' '
        return  ret

    def add_neigh(self, node, weight):
        # Adds nodes to the neighs list of the node
        # INPUT:
        #   node: object Node
        #   weight: weight of the edge
        self.neighs.append((node, float(weight)))

class Graph(Node):
    # Class that creates a graph
    def __init__(self, sparse_mat):
        # Class constructor
        # INPUT:
        #   sparse_mat: np.array with [node A, node B, weight] as rows

        self.nodes = []
        self.sparse = sparse_mat

        for row in sparse_mat:
            node_A = row[0]
            node_B = row[1]
            weight = row[2]

            A_in_graph = self.node_in_graph(node_A)
            B_in_graph = self.node_in_graph(node_B)

            if A_in_graph == False:
                self.nodes.append(Node(node_A))
            
                if B_in_graph == False:
                    self.nodes.append(Node(node_B))
                    self.nodes[-2].add_neigh(self.nodes[-1], weight)
                    self.nodes[-1].add_neigh(self.nodes[-2], weight)

                else: # B_in_graph -> index where node B is in self.nodes
                    self.nodes[-1].add_neigh(self.nodes[B_in_graph-1], weight)
                    self.nodes[B_in_graph-1].add_neigh(self.nodes[-1], weight)
                
            else:
                if B_in_graph == False:
                    self.nodes.append(Node(node_B))
                    self.nodes[-1].add_neigh(self.nodes[A_in_graph-1], weight)
                    self.nodes[A_in_graph-1].add_neigh(self.nodes[-1], weight)

                else:
                    self.nodes[A_in_graph-1].add_neigh(self.nodes[B_in_graph-1], weight)
                    self.nodes[B_in_graph-1].add_neigh(self.nodes[A_in_graph-1], weight)

    def node_in_graph(self, node_id):
        # determines whether or not a node is in a graph. Looks for it in self.nodes
        # INPUT:
        #   node_id: int that specifies the node to look for
        # OUTPUT:
        #   returns index of the node in the list if exists, if not return False
        for idx, node in enumerate(self.nodes):
            if node.id == node_id: 
                return idx+1
        return False

    def _min_dist(self, nodes):
        # determines the minimum distance of a list of nodes
        # INPUT:
        #   nodes: list of nodes to look on
        # OUTPUT:
        #   min_node: the node with the minimum distance
        min_node = nodes[0]
        min_dist = min_node.distance
        for i in range(1, len(nodes)):
            if nodes[i].distance < min_dist:
                min_dist = nodes[i].distance
                min_node = nodes[i]
        return min_node

    def sparse2adj(self, sparse):
        # transforms a sparse matrix into a adjacency matrix
        # INPUT:
        #   sparse: numpy array - sparse matrix of graph
        # OUTPUT:
        #   adj: numpy array - adjacency matrix of graph
        adj = np.zeros((len(self.nodes),len(self.nodes)))
        for row in sparse:
            node_A = int(row[0])
            node_B = int(row[1])
            weight = float(row[2])
            adj[node_A][node_B] = weight
            adj[node_B][node_A] = weight
        return adj

    def draw_graph(self):
        # draws each node and its corresponding edges
        G = nx.convert_matrix.from_numpy_array(self.sparse2adj(self.sparse))
        pos = nx.spring_layout(G)
        labels = {}
        for i in self.sparse:
            labels.setdefault((i[0],i[1]),i[2])
        plt.title("Graph visualization")
        nx.draw(G, pos, with_labels = True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
        plt.show()
        
    def routing_table(self):
        # determines and prints the routing table the graph
        # OUTPUT:
        #   table: DataFrame of the routing table of graph
        id = []
        dist = []
        pred = []
        for node in self.nodes:
            id.append(node.id)
            dist.append(node.distance)
            pred.append(node.predecessor)
        table = np.column_stack((id,dist,pred))
        table = pd.DataFrame(table)
        table.columns = ["id", "distance", "predecessor"]
        print("\n ----------------- Routing Table ---------------------- \n")
        print(table.to_string(index=False))
        return table

    def draw_tree(self, route_table):
        # draws the tree corresponding to the routing table
        # INPUT:
        #   routing_table: DataFrame with the routing table
        route_m = route_table.to_numpy()
        route_m[:, [2, 1]] = route_m[:, [1, 2]] 
        route_m = np.delete(route_m, (0), axis=0)
        G = nx.convert_matrix.from_numpy_array(self.sparse2adj(route_m))
        plt.title("Final Tree")
        nx.draw(G, with_labels = True)
        plt.show()
    
    def dijkstra(self, s):
        # Performs Dijkstra's algorithm and plots the original graph and the resulting tree

        # starting time
        start = time.time()
        for ind in range(len(self.nodes)):
            if self.nodes[ind].id == s:
                break
        actual = self.nodes[ind]
        actual.distance = 0
        actual.predecessor = -1
        unvisited = copy.copy(self.nodes)

        it = 0
        print("\n -------------- Execution Time per Iteration ----------- \n")

        while len(unvisited) > 0:
            it += 1
            # iteration start time
            start_i = time.time()
            for neigh in actual.neighs:
                if neigh[0] in unvisited: #se puede crear un atributo
                    temp_dist = actual.distance + neigh[1]
                    if temp_dist < neigh[0].distance:
                        neigh[0].distance = temp_dist
                        neigh[0].predecessor = actual.id
            unvisited.remove(actual)
            if len(unvisited) != 0:
                actual = self._min_dist(unvisited)
            # iteration end time
            end_i = time.time()
            print("Runtime of iteration {0} in Dijkstra's Algorithm is {1:.50f}".format(it, end_i - start_i))

        # end time
        end = time.time()
        # total time taken
        print("\n ----------------- Execution Time ---------------------- \n")

        print(f"Runtime of Dijkstra's Algorithm is {end - start}")

        print("\n ----------------- Number of Iterations ---------------- \n")
        print("Dijkstra took {0} iterations to finish".format(it))

        self.draw_tree(self.routing_table())

def create_traffic_matrix(nodos, t_max = 1):
    #Crea una matriz de tráfico en la red con d_(i,j)=cantidad de tráfico de el nodo i al nodo j
    #Input:
    #   nodos: Numero de nodos del grafo
    #   t_max: Trafico máximo en la red
    #Output:
    #   mT: Matriz de tráfico de la red con el tráfico elegido de una función uniforme entre 0 y t_max
    
    mT = np.zeros((nodos, nodos))
    for i in range(nodos):
        for j in range(nodos):
            if i != j:
                mT[i,j] = np.random.uniform(0, t_max)
    return mT

def create_PL_vector(sparse, nodos):
    #Crea el vector que se introducirá al PL para encontrar la división óptima de tráfico en la red
    #Ademas, como este es un vector de unos, crea la codificación adecuada para poder saber que significa cada variable en el output
    #Input:
    #   nodos: Numero de nodos en la red
    #   sparse: matriz de adyacenia y pesos del grafo
    #Output:
    #   c: Vector de unos a introducir en el PL, cada 1 es la cantidad de tráfico dirigido a t que pasa por i,j
    #   code: Vector de codigo, una tupla de forma (i,j,t)
    c = []
    code = []
    for i in sparse:
        for j in range(nodos):
            if j != i[0]:
                c.append(1)
                code.append((i[0],i[1],j))
            if j != i[1]:
                c.append(1)
                code.append((i[1],i[0],j))
    return c, code

def create_PL_eq_restrictions(nodos, code, traffic):
    #Función encargada de crear la matriz y el vector de restricciones de igualdad para el PL
    #Input: 
    #   nodos: Número de nodos de la red
    #   code: vector de codificación de las variables Y_ij^t
    #   traffic: matriz de trafico en la red
    #Output:
    #   A_eq: Matriz de restricciones de igualdad
    #   b_eq: vector de restricciones de igualdad
    #   Para llevar a cabo las restricciones de forma A_eq*x=b_eq

    D_t = []
    for i in np.transpose(traffic):
        D_t.append(sum(i))

    A_eq = np.zeros((nodos**2,len(code)))
    b_eq = []
    for i in range(nodos):
        for t in range(nodos):
            for k in range(len(code)):
                if code[k][1] == i and code[k][2] == t:
                    A_eq[i*nodos+t, k] = 1
                elif code[k][0] == i and code[k][2] == t:
                    A_eq[i*nodos+t, k] = -1
            if i==t:
                b_eq.append(D_t[i])
            else:
                b_eq.append(-traffic[i,t])
    return A_eq,b_eq

def create_PL_ineq_restrictions(nodos, code, sparse):
    #Función encargada de crear la matriz y el vector de restricciones de desigualdad para el PL
    #Input: 
    #   nodos: Número de nodos de la red
    #   code: vector de codificación de las variables Y_ij^t
    #   sparse: matriz de adyacencia con pesos
    #Output:
    #   A_ineq: Matriz de restricciones de desigualdad
    #   b_ineq: vector de restricciones de desigualdad
    #   Para llevar a cabo las restricciones de forma A_eq*x<=b_eq

    A_ineq = np.zeros((len(sparse)*2,len(code)))
    b_ineq = []
    for i in range(len(sparse)):
        for k in range(len(code)):
            if sparse[i][0] == code[k][0] and sparse[i][1] == code[k][1]:
                A_ineq[i, k] = 1
            elif sparse[i][1] == code[k][0] and sparse[i][0] == code[k][1]:
                A_ineq[i+1, k] = 1
        b_ineq.append(sparse[i][2])
        b_ineq.append(sparse[i][2])
    return A_ineq,b_ineq

def solve_lin_prob(c, A_eq, b_eq, A_ineq, b_ineq, nodos):
    #Funcion encargada de ejecutar el problema lineal
    #res = opt.linprog(c=c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq[0:nodos**2-nodos], b_eq=b_eq[0:nodos**2-nodos])
    res = opt.linprog(c=c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq)
    return res
    
def create_dual_vector(sparse, nodos, traffic):
    

    D_t = []
    for i in traffic:
        D_t.append(sum(i))

    D_vec = []
    code2 = []
    for t in range(nodos):
        for i in range(nodos):
            if i==t:
                D_vec.append(D_t[t])
            else:
                D_vec.append(-traffic[i, t])
            code2.append((1, i, t))
    for k in sparse:
        D_vec.append(k[2])
        code2.append((2, k[0], k[1]))
    
    return D_vec, code2

def create_dual_eq_restrictions(code2, nodos):

    Ad_eq = np.zeros((nodos, len(code2)))
    bd_eq = [0 for i in range(nodos)]
    c = 0
    for i in range(len(code2)):
        if code2[i][0] == 1:
            if code2[i][1] == code2[i][2]:
                Ad_eq[c,i] = 1
                c += 1
    return Ad_eq, bd_eq

def create_dual_ineq_restrictions(code2, nodos, sparse):

    Ad_ineq = np.zeros(((2*nodos)*len(sparse) + len(sparse), len(code2)))
    bd_ineq = []
    c=0
    for i in range(nodos**2, nodos**2+len(sparse)):
        for t in range(nodos):
            for k in range(0, nodos**2):
                if code2[k][1] == code2[i][1] and code2[k][2] == t:
                    Ad_ineq[c, k] = 1
                elif code2[k][1] == code2[i][2] and code2[k][2] == t:
                    Ad_ineq[c, k] = -1
            Ad_ineq[c, i] = -1
            bd_ineq.append(1)
            c += 1
            for k in range(0, nodos**2):
                if code2[k][1] == code2[i][2] and code2[k][2] == t:
                    Ad_ineq[c, k] = 1
                elif code2[k][1] == code2[i][1] and code2[k][2] == t:
                    Ad_ineq[c, k] = -1
            Ad_ineq[c, i] = -1
            bd_ineq.append(1)
            c += 1
    for i in range(nodos**2, nodos**2+len(sparse)):
        Ad_ineq[c, i] = -1
        bd_ineq.append(0) 
        c += 1
    return Ad_ineq, bd_ineq

def solve_dual_prob(c, A_eq, b_eq, A_ineq, b_ineq):
    #Funcion encargada de ejecutar el problema dual
    res = opt.linprog(c=c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq)
    return res
    
def change_sparse(sparse, Z, nodos):
    #Funcion encargada de asignar los nuevos pesos al grafo para lograr un tráfico óptimo
    new_sparse = [[sparse[0,0], sparse[0,1], 1+Z[0]]]
    for i,j in zip(range(nodos**2+1, nodos**2+len(sparse)), range(len(sparse))):
        #print(sparse[j,0], sparse[j,1])
        tmp = [sparse[j,0], sparse[j,1], 1+Z[i]]
        new_sparse.append(tmp)
    #print(new_sparse)
    return new_sparse

def create_Knm(nodos, K, Y, traffic):

    #Retorna un diccionario con Kmn, xi y fkm
    Knm = {}
    for n in range(nodos):
        for m in range(nodos):
            if n!=m:
                key = (n,m)
                Knm[key] = [[], [], []]
                for k in range(len(K)):
                    if K[k][0] == n and K[k][2] == m:
                        n_val = Knm.get(key)
                        n_val[0].append(K[k][1])
                        n_val[1].append(Y[k])
                        Knm[key] = n_val
    for key in Knm:
        elem = Knm[key]
        for j in elem[0]:
            if j!= key[1]:
                next_hop = Knm[(j, key[1])]
                elem[2].append(sum(next_hop[1]))
            else:
                elem[2].append(0)
            Knm[key] = elem
    return Knm
                     
def min_max (n, m, Knm):
    arr = Knm[(n,m)]
    num = len(arr[1])
 
    # Traverse through all array elements
    for i in range(num-1):
    # range(n) also work but outer loop will
    # repeat one time more than needed.
 
        # Last i elements are already in place
        for j in range(0, num-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[1][j] < arr[1][j + 1] :
                arr[0][j], arr[0][j + 1] = arr[0][j + 1], arr[0][j]
                arr[1][j], arr[1][j + 1] = arr[1][j + 1], arr[1][j]
                arr[2][j], arr[2][j + 1] = arr[2][j + 1], arr[2][j]
    i = 0
    l = [0]
    max_arr = [[], []]
    for i in range(len(arr[0])):
        #print(arr[1][i])
        max_arr[0].append(arr[0][i])
        max_arr[1].append(((l[-1] + arr[2][i]/(i+1))/arr[1][i]))
    if len(max_arr[0]) > 2:
        max_arr[0] = max_arr[0][0:2]
        max_arr[1] = max_arr[1][0:2]
    print(n, m)
    print(max_arr)
    


def create_random_sparse(nodos):
    # Creates a random sparse matrix given a number of nodes
    # INPUT:
    #   nodos: The number of nodes in the graph
    # OUTPUT:
    #   mA: numpy array of the sparse matriz
    nodos = int(nodos)
    
    #Creo una lista con todos los posibles pares para elegir
    choice = []
    for i in range(nodos):
        for j in range(i+1, nodos):
            choice.append([i,j, rd.randint(1, 500)])


    i = 0
    while i <= nodos:
        tmp = rd.choice(choice)
        choice.remove(tmp)
        tmp = np.array([tmp])
        
        if i == 0:
            mA = tmp
        else:
            mA = np.concatenate((mA, tmp), axis=0)
        i+=1
    i = 0
    while i<int(nodos/2):
        tmp = rd.choice(choice)
        choice.remove(tmp)
        tmp = np.array([tmp])
        mA = np.concatenate((mA, tmp), axis=0)
        i+=1
    for i in range(nodos):
        var = False
        for j in mA:
            if i==j[0] or i==j[1]:
                var = True
                break
        if not var:
            for j in choice:
                if i==j[0] or i==j[1]:
                    tmp = np.array([j])
                    mA = np.concatenate((mA, tmp), axis=0)
    return mA

def import_csv(filename):
    # Reads a csv and converts to numpy array
    # INPUT:
    #   filename: name of the csv file
    # OUTPUT:
    #   numpy array of the csv file
    df = pd.read_csv(filename, sep=';')
    return df.to_numpy()

def import_txt(filename):
    # Reads a txt and converts to numpy array
    # INPUT:
    #   filename: name of the txt file
    # OUTPUT:
    #   numpy array of the txt file
    f = open(filename, "r")
    first1 = True
    text = f.readline()
    while text != '':
        a = ''
        b = ''
        c = ''
        start = 'a'
        for i in text:
            if i != ' ' and start == 'a':
                a+=i
            elif i!= ' ' and start == 'b':
                b+=i
            elif i != '\n' and start == 'c':
                c+=i
            else:
                if start == 'a':
                    start = 'b'
                elif start == 'b':
                    start = 'c'
        if first1:
            mT = np.array([[int(a),int(b),int(c)]])
            first1 = False
        else:
            tmp = np.array([[int(a),int(b),int(c)]])
            mT = np.concatenate((mT, tmp), axis=0)
        text = f.readline()
    return mT
    
