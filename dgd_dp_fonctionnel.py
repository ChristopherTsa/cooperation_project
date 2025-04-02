import math as m
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from centralized_solution import solve

# --- Paramètres Globaux ---
sigma = 0.5
nu = 1.0

# --- Fonctions Utilitaires ---
def load_data(filename):
    """Charge les données à partir d'un fichier pickle."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_data(x_data, y_data):
    """Visualise les données x_data et y_data sous forme de scatter plot."""
    plt.figure()
    plt.scatter(x_data, y_data, label='Données', s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Visualisation des données chargées")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/data_visualization.pdf")
    plt.show()

def euclidean_kernel(x, xi):
    """Calcule le kernel Euclidien."""
    return np.exp(-np.linalg.norm(x - xi)**2)

def kernel_matrix(X, X_prime):
    """Calcule la matrice de kernel K[i,j] = k(X[i], X_prime[j])."""
    K = np.zeros((len(X), len(X_prime)))
    for i, x in enumerate(X):
        for j, x_prime in enumerate(X_prime):
            K[i, j] = euclidean_kernel(x, x_prime)
    return K

def nystrom_approximation(X, m):
    """Sélectionne aléatoirement m points pour l'approximation de Nyström."""
    n = len(X)
    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    X_selected = [X[i] for i in ind]
    M_indices = ind
    return X_selected, M_indices

def create_weighted_graphs(num_agents):
    """Crée différentes structures de graphes avec des poids appropriés pour les algorithmes décentralisés."""
    graphs = []
    graph_names = []
    
    # 1. Line Graph (Path Graph)
    line_graph = nx.path_graph(num_agents)
    graph_names.append("Line Graph")
    graphs.append(line_graph)
    
    # 2. Small-world Graph (Watts-Strogatz)
    # Each node connected to k nearest neighbors with probability p of rewiring edges
    k = min(2, num_agents-1)
    p = 0.3
    small_world = nx.watts_strogatz_graph(num_agents, k, p)
    graph_names.append("Small-world Graph")
    graphs.append(small_world)
    
    # 3. Fully Connected Graph (Complete Graph)
    fully_connected = nx.complete_graph(num_agents)
    graph_names.append("Fully Connected Graph")
    graphs.append(fully_connected)
    
    for graph in graphs:
        for i in range(num_agents):
            neighbors = list(graph.neighbors(i))
            for j in neighbors:
                weight = 1.0 / (max(graph.degree(i), graph.degree(j)) + 1)
                graph[i][j]['weight'] = weight
    
    return graphs, graph_names

# --- Fonctions de Visualisation ---
def plot_convergence(iterations, optimality_gap, algorithm_name):
    """Affiche le graphique de convergence (optimality gap vs itérations)."""
    plt.figure()
    plt.loglog(iterations, optimality_gap)
    plt.xlabel("Iterations")
    plt.ylabel("Optimality Gap (||alpha_t - alpha*||)")
    plt.title(f"Convergence de {algorithm_name}")
    plt.grid(True)
    plt.savefig(f"results/convergence/convergence_{algorithm_name}.pdf")
    plt.show()

def visualize_function(x_prime, alpha, X_m_points, algorithm_name, y_nystrom, selected_data=None):
    """Visualise la fonction apprise sur une grille uniforme et affiche les vraies valeurs y pour les points Nyström."""
    nt = len(x_prime)
    f_x_prime = np.zeros(nt)
    for i in range(nt):
        for j in range(len(X_m_points)):
            f_x_prime[i] += alpha[j] * euclidean_kernel(x_prime[i], X_m_points[j])

    plt.figure()
    plt.plot(x_prime, f_x_prime, label='Fonction Apprise')
    plt.scatter(X_m_points, y_nystrom, color='red', marker='x', label='Points Nyström')
    if selected_data is not None:
        x_selected, y_selected = selected_data
        plt.scatter(x_selected, y_selected, color='green', marker='o', label='Points sélectionnés', alpha=0.5)
    plt.title(f"Fonction apprise avec {algorithm_name}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/learned_function/learned_fonction_{algorithm_name}.pdf")
    plt.show()

def plot_reconstruction_multi(x_prime, methods_alpha, X_m_points, y_nystrom, selected_data=None):
    """Affiche sur un même graphe la reconstruction des fonctions apprises pour chaque méthode."""
    plt.figure()
    for method_name, alpha in methods_alpha.items():
        f_x = np.zeros(len(x_prime))
        for i in range(len(x_prime)):
            for j in range(len(X_m_points)):
                f_x[i] += alpha[j] * euclidean_kernel(x_prime[i], X_m_points[j])
        plt.plot(x_prime, f_x, label=f"{method_name}")
    plt.scatter(X_m_points, y_nystrom, color='red', marker='x', label="Points Nyström")
    if selected_data is not None:
        x_selected, y_selected = selected_data
        plt.scatter(x_selected, y_selected, color='green', marker='o', label="100 points sélectionnés", alpha=0.5)
    plt.title("Reconstruction comparée des 4 méthodes")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/learned_function/comparison_reconstruction.pdf")
    plt.show()

def plot_convergence_multi(iterations, convergence_data):
    """
    Affiche sur un même graphe les courbes de convergence (optimality gap) pour chaque méthode itérative."""
    plt.figure()
    for method_name, gap in convergence_data.items():
        plt.loglog(iterations, gap, label=f"{method_name}")
    plt.xlabel("Itérations")
    plt.ylabel("Optimality Gap")
    plt.title("Convergence comparée des 4 méthodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/convergence/comparison_convergence.pdf")
    plt.show()
    

def dgd_dp(
    agents_data_indices,
    agents_x_data,
    agents_y_data,
    X_m_points,
    Kmm,
    communication_graph,
    step_size,
    num_iterations,
    alpha_star_centralized,
    epsilon,
    verbose):
    """Implémentation de la Descente de Gradient Décentralisée avec bruit laplacien (DGD-DP)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))
    
    W = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        neighbors = list(communication_graph.neighbors(i))
        for j in neighbors:
            W[i, j] = communication_graph.get_edge_data(i, j)['weight']

    optimality_gap_history_dgd_dp = []
    # Suivi de l'optimality gap par agent
    optimality_gap_by_agent_dgd_dp = [[] for _ in range(num_agents)]

    for iteration in range(num_iterations):
        gamma_k = 1 / (1 + 0.001 * (iteration ** 0.9))  # Facteur d'atténuation
        alpha_k = 0.002 / (1 + 0.001 * iteration)  # Learning rate décroissant
        nu_k = (0.01/epsilon) * (1/(1+0.001*iteration**0.1)) # Variance du bruit laplacien
        
        new_agent_alphas = [np.zeros(m) for _ in range(num_agents)]
        
        for agent_id in range(num_agents):
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent
            noise = np.random.laplace(0, nu_k, size=grad_local.shape) 
            
            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id))
            for neighbor_id in neighbors:
                alpha_avg += gamma_k * W[agent_id, neighbor_id] * ((agent_alphas[neighbor_id] + noise) - alpha_agent)
            new_agent_alphas[agent_id] = alpha_agent + alpha_avg - alpha_k * grad_local
            
            # Calculer et enregistrer l'optimality gap pour chaque agent
            optimality_gap_agent = np.linalg.norm(alpha_agent - alpha_star_centralized)
            optimality_gap_by_agent_dgd_dp[agent_id].append(optimality_gap_agent)
            
        agent_alphas = new_agent_alphas
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgd_dp.append(optimality_gap)
        if verbose:
            print(f"DGD - Iteration {iteration+1}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")

    return agent_alphas, optimality_gap_history_dgd_dp


if __name__ == "__main__":
    print("--- 0. Préparation des données ---")
    # Choisir n, m, a et diviser les données
    n_total = 100
    m_nystrom = m.ceil(m.sqrt(n_total))
    num_agents = 5
    nt = 250
    points_per_agent = n_total // num_agents
    
    print("--- 0.1. Charger les données ---")
    data_part1 = load_data('data/first_database.pkl')
    x_data, y_data = data_part1[0], data_part1[1]
    print(f"Taille de x_data: {len(x_data)}")
    print(f"Taille de y_data: {len(y_data)}")
    # --- Visualisation des données ---
    #print("Visualisation des données...")
    #visualize_data(x_data, y_data)
    #print("Visualisation des données terminée. Graphique sauvegardé dans data_visualization.pdf")

    print("--- 0.2. Attribution des données aux agents ---")
    all_indices = list(range(n_total))
    np.random.shuffle(all_indices)
    agents_data_indices = [all_indices[i*points_per_agent:(i+1)*points_per_agent] for i in range(num_agents)]
    print(f"Indices des données pour chaque agent:")
    for (i, indices) in enumerate(agents_data_indices):
        print(f"Agent {i}: {indices}")
    agents_x_data = [[x_data[i] for i in indices] for indices in agents_data_indices]
    agents_y_data = [[y_data[i] for i in indices] for indices in agents_data_indices]
    
    print("--- 1.0 Initialisation du graphe de communication ---")
    graphs, graph_names = create_weighted_graphs(num_agents)

    print("--- 0.3 Calcul de la solution centralisée ---")
    alpha_star_centralized, M_indices = solve(x_data[:n_total], y_data[:n_total], selection=True)
    print(f"Indices des points sélectionnés pour l'approximation de Nyström: {M_indices}")
    X_m_points = [x_data[i] for i in M_indices]
    y_nystrom = [y_data[i] for i in M_indices]
    Kmm = kernel_matrix(X_m_points, X_m_points)
    print("Solution centralisée calculée.")
    x_prime_grid = np.linspace(-1, 1, nt)
    #visualize_function(x_prime_grid, alpha_star_centralized, X_m_points, "Centralized",  y_nystrom,(x_data[:n_total], y_data[:n_total]))

        
    print("--- 1. Méthodes Distribuées Classiques (DGD, GT, DD, ADMM) ---")
    num_iterations = 10000
    
    print("--- 1.0 Initialisation du graphe de communication ---")
    graphs, graph_names = create_weighted_graphs(num_agents)
    #results_by_graph = {}
    
    for graph_idx, (graph, graph_name) in enumerate(zip(graphs, graph_names)):
        communication_graph = graph
        
    print("--- 3. DGD-DP ---")
    communication_graph_dp = communication_graph.copy() # Réutiliser le graphe de DGD
    step_size_dgd_dp = 0.01
    num_iterations_dgd_dp = 10000
    noise_std_dp = [0.1, 1, 10] # Écart-type du bruit laplacien - à calibrer pour epsilon
    
    agent_alphas_dgd_dp_01, optimality_gap_history_dgd_dp01 = dgd_dp(
            agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dp,
            step_size_dgd_dp, num_iterations_dgd_dp, alpha_star_centralized, 0.1, True)
        

    agent_alphas_dgd_dp_1, optimality_gap_history_dgd_dp_1 = dgd_dp(
            agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dp,
            step_size_dgd_dp, num_iterations_dgd_dp, alpha_star_centralized, 1, True
        )
    
    agent_alphas_dgd_dp_10, optimality_gap_history_dgd_dp_10 = dgd_dp(
            agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dp,
            step_size_dgd_dp, num_iterations_dgd_dp, alpha_star_centralized, 10, True
        )
    
    min_length = min(len(optimality_gap_history_dgd_dp01), 
                        len(optimality_gap_history_dgd_dp_1),
                        len(optimality_gap_history_dgd_dp_10))
    
    convergence_data_dgd = {
        'eps = 0.1' : optimality_gap_history_dgd_dp01[:min_length],
        'eps = 1' : optimality_gap_history_dgd_dp_1[:min_length],
        'eps = 10' : optimality_gap_history_dgd_dp_10[:min_length]
    }
    
    iterations = range(min_length)
    plot_convergence(iterations, optimality_gap_history_dgd_dp_10, "DGD-DP eps = 10")
    plot_convergence_multi(iterations, convergence_data_dgd)
    
    
    
    print("--- Programme terminé ---")
