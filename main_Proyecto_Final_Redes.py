# Redes de Computadores 2022
# Universidad del Rosario - School of Engineering, Science and Technology
# Pr. David Celeita
from Proyecto_Final_Redes import *
import copy

# This is 0 main function. The interaction between user and tool si performed here.

print("\n Welcome to Link-State routing Tool based on Dijkstra!")

is_valid = False
while is_valid != True:
    num_nodes = int(input("Choose the number of nodes in the graph. They must be between 10 and 100: "))
    if 10 <= num_nodes and num_nodes <= 100:
        is_valid = True


rand_sparse = create_random_sparse(num_nodes)
print("matriz de adyacencia creada")
traffic = create_traffic_matrix(num_nodes)
print("matriz de tráfico creada")
graph = Graph(rand_sparse)
graph.draw_graph()
source = int(input('Desde que nodo desea enviar un paquete?'))
t = int(input('Hacia que nodo desea enviar el paquete?'))
print(f'El tráfico que va hacia {t} desde {source} tiene un valor de {traffic[source][t]}')
graph.dijkstra(source)

start_problems = time.time()
c, code = create_PL_vector(rand_sparse, num_nodes)
print("Vector del PL creado")
A_eq, b_eq = create_PL_eq_restrictions(num_nodes, code, traffic)
print("Restricciones de igualdad del PL creadas")
A_ineq, b_ineq = create_PL_ineq_restrictions(num_nodes, code, rand_sparse)
print("Restricciones de desigualdad del PL creadas")
Y = solve_lin_prob(c, A_eq, b_eq, A_ineq, b_ineq)
print("PL solucionado con resultado:")
print(Y.success)
print(Y.x)
D_vec, code2 = create_dual_vector(rand_sparse, num_nodes, traffic)
print("Vector del problema dual creado")
Ad_eq, bd_eq = create_dual_eq_restrictions(code2, num_nodes)
print("Restricciones de igualdad del dual creadas")
Ad_ineq, bd_ineq = create_dual_ineq_restrictions(code2, num_nodes, rand_sparse)
print("Restricciones de desigualdad del dual creadas")
Z = solve_dual_prob(c=D_vec, A_eq=Ad_eq, b_eq = bd_eq, A_ineq=Ad_ineq, b_ineq=bd_ineq)
print("Problema dual resuelto con resultado:")
print(Z.success)
print(Z.x)
end_problems = time.time()
print(f"el tiempo de solución de los problemas lineal y dual es de {end_problems-start_problems}")

start_heuristic = time.time()
optimal_sparse = change_sparse(rand_sparse, Z.x, num_nodes)
graph = Graph(optimal_sparse)
graph.draw_graph()
graph.dijkstra(source)
Knm = create_Knm(num_nodes, code, Y.x)
next_hops = min_max(source,t,Knm)[0]
end_heuristic = time.time()
print(f"el tiempo de implementación de la heurística es de {end_heuristic-start_heuristic}")

print(f"el tiempo de implementación de la solución es de {end_problems-start_problems+end_heuristic-start_heuristic}")
print(f'Si desea enviar el paquete a {t} Se deceria distribuir el tráfico entre los routers {next_hops}')
