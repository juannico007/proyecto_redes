# Redes de Computadores 2022
# Universidad del Rosario - School of Engineering, Science and Technology
# Pr. David Celeita
from Proyecto_Final_Redes import *

# This is 0 main function. The interaction between user and tool si performed here.

print("\n Welcome to Link-State routing Tool based on Dijkstra!")

selection = input("Select 0 if you want me to create a random graph for you. Press 1 if you want to upload it \n Type your selection and press enter: ")

if selection == '0':
    is_valid = False
    while is_valid != True:
        num_nodes = int(input("Choose the number of nodes in the graph. They must be between 15 and 50: "))
        if 3 <= num_nodes and num_nodes <= 500:
            is_valid = True


    rand_sparse = create_random_sparse(num_nodes)
    print("matriz de adyacencia creada")
    traffic = create_traffic_matrix(num_nodes)
    print("matriz de tráfico creada")
    c, code = create_PL_vector(rand_sparse, num_nodes)
    print("Vector del PL creado")
    A_eq, b_eq = create_PL_eq_restrictions(num_nodes, code, traffic)
    print("Restricciones de igualdad del PL creadas")
    A_ineq, b_ineq = create_PL_ineq_restrictions(num_nodes, code, rand_sparse)
    print(len(c), len(A_eq), len(A_ineq))
    print("Restricciones de desigualdad del PL creadas")
    graph = Graph(rand_sparse)
    graph.draw_graph()
    graph.dijkstra()
    Y = solve_lin_prob(c, A_eq, b_eq, A_ineq, b_ineq, num_nodes)
    print("PL solucionado")
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
    optimal_sparse = change_sparse(rand_sparse, Z.x, num_nodes)
    graph = Graph(optimal_sparse)
    graph.draw_graph()
    graph.dijkstra()

elif selection == '1':
    is_valid = False
    while is_valid != True:
        doc_type = str(input("If you want to upload a CSV file type CSV or if you want a TXT file type TXT. \n Type your selection and press enter: "))
        if doc_type == "CSV" or doc_type == "TXT":
            is_valid = True
    
    filename = input("\n Now, type the name of the file. Remember that it must be in the same folder of this .exe or specify the whole path. \n Type you filename with the extension: ")
    
    if doc_type == "CSV":
        sparse_m = import_csv(filename)
        graph = Graph(sparse_m)
        graph.draw_graph()
        graph.dijkstra()

    else:
        sparse_m = import_txt(filename)
        graph = Graph(sparse_m)
        graph.draw_graph()
        graph.dijkstra()






