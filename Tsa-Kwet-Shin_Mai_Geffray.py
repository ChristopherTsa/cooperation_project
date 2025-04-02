import math as m
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
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

# --- Fonctions pour les graphes ---
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
        # Add self-loops first
        for i in range(num_agents):
            graph.add_edge(i, i)
        
        # Step 1: Create unnormalized weights using Metropolis-Hastings rule
        for i in range(num_agents):
            neighbors = list(graph.neighbors(i))
            for j in neighbors:
                if i != j:
                    weight = 1.0 / (max(graph.degree(i), graph.degree(j)) - 1)
                    graph[i][j]['weight'] = weight
        
        # Step 2: Normalize self-loops to make row sums equal to 1
        for i in range(num_agents):
            neighbors = list(graph.neighbors(i))
            neighbor_sum = sum(graph[i][j]['weight'] for j in neighbors if i != j)
            graph[i][i]['weight'] = 1.0 - neighbor_sum

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
    plt.xlabel("Number of iterations")
    plt.ylabel(r"Optimality gap $\|\alpha_i - \alpha^*\|$")
    plt.title("Convergence comparée des 4 méthodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/convergence/comparison_convergence.pdf")
    plt.show()

def plot_convergence_by_agent(iterations, optimality_gap_by_agent, algorithm_name, communication_graph=None, step_size=None):
    """
    Affiche les courbes de convergence (optimality gap) pour chaque agent.
    
    Parameters:
    - iterations: L'axe x (numéros d'itération)
    - optimality_gap_by_agent: Une liste où chaque élément est l'historique d'optimality gap d'un agent
    - algorithm_name: Le nom de l'algorithme
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for agent_id, gap_history in enumerate(optimality_gap_by_agent):
        ax.plot(iterations, gap_history, label=f'agent {agent_id + 1}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"Optimality gap $\|\alpha_i - \alpha^*\|$")
    
    ax.set_title(f"Convergence by agent for {algorithm_name}")
    ax.legend()
    ax.grid(True, which='major')
    
    if communication_graph is not None:
        graph = communication_graph.copy()
        # Afficher la matrice de poids W
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        num_agents = len(optimality_gap_by_agent)
        W = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            neighbors = list(graph.neighbors(i))
            for j in neighbors:
                W[i, j] = graph.get_edge_data(i, j)['weight']
        
        def matrix_to_latex(W):
            latex = r"$W = \begin{bmatrix}"
            for i in range(W.shape[0]):
                row = " & ".join([
                    f"{W[i, j]:.2f}" if W[i, j] != 0 else "0" for j in range(W.shape[1])
                ])
                latex += row + r" \\ "
            latex += r"\end{bmatrix}$"
            return latex

        latex_W = matrix_to_latex(W)
        ax.text(0.75, 0.15, latex_W, fontsize=14,
                transform=ax.transAxes, ha='right')
        
        for i in graph.nodes:
            graph.remove_edge(i, i)
        
        # Afficher le graphe de communication
        pos = nx.circular_layout(graph)
        inset_ax = fig.add_axes([0.15, 0.15, 0.2, 0.2])  # [left, bottom, width, height]
        nx.draw(graph, pos, ax=inset_ax, node_size=50, with_labels=False, node_color='black')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
    
    # Afficher le step size
    if step_size is not None:
        ax.text(0.2, 0.5, rf"$s = {step_size}$", fontsize=20,
            transform=ax.transAxes, ha='right')
    
    #plt.tight_layout()
    plt.savefig(f"results/convergence/convergence_by_agent_{algorithm_name}.pdf")
    plt.show()

def plot_convergence_agents_multi(iterations, methods_convergence_data):
    """
    Affiche sur un même graphe les courbes de convergence pour chaque agent de chaque méthode.
    
    Parameters:
    - iterations: L'axe x (numéros d'itération)
    - methods_convergence_data: Un dictionnaire où chaque clé est le nom d'une méthode et
                               chaque valeur est une liste des historiques d'optimality gap pour chaque agent
    """
    plt.figure(figsize=(12, 8))
    
    # Définir les couleurs pour chaque méthode
    method_colors = {
        "DGD": "blue",
        "GradientTracking": "red",
        "DualDecomposition": "green",
        "ADMM": "purple"
    }
    
    for method_name, agent_gaps in methods_convergence_data.items():
        color = method_colors.get(method_name, "gray")
        
        for agent_id, gap_history in enumerate(agent_gaps):
            
            # Le premier agent de chaque méthode apparaît dans la légende pour la méthode
            if agent_id == 0:
                plt.loglog(iterations, gap_history, color=color,
                         label=f"{method_name}", alpha=0.7)
            else:
                plt.loglog(iterations, gap_history, color=color, alpha=0.7)
    
    plt.xlabel("Number of iterations")
    plt.ylabel(r"Optimality gap $\|\alpha_i - \alpha^*\|$")
    plt.title("Convergence by agent for all methods")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/convergence/comparison_convergence_by_agent.pdf")
    plt.show()

# --- Fonctions pour les Algorithmes Distribués ---
# --- Algorithmes de la partie 1 ---
def decentralized_gradient_descent(
    agents_data_indices,
    agents_x_data,
    agents_y_data,
    X_m_points,
    Kmm,
    communication_graph,
    step_size,
    num_iterations,
    alpha_star_centralized,
    verbose=False):
    """Implémentation de la Descente de Gradient Décentralisée (DGD)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    
    agents_Knm = [kernel_matrix(agents_x_data[i], X_m_points) for i in range(num_agents)]
    
    W = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        neighbors = list(communication_graph.neighbors(i))
        for j in neighbors:
            W[i, j] = communication_graph.get_edge_data(i, j)['weight']

    optimality_gap_history_dgd = []
    # Suivi de l'optimality gap par agent
    optimality_gap_by_agent_dgd = [[] for _ in range(num_agents)]

    for iteration in range(num_iterations):
        new_agent_alphas = [np.zeros(m) for _ in range(num_agents)]
        for agent_id in range(num_agents):
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent

            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id))
            for neighbor_id in neighbors:
                alpha_avg += W[agent_id, neighbor_id] * agent_alphas[neighbor_id]
            new_agent_alphas[agent_id] = alpha_avg - step_size * grad_local
            
            # Calculer et enregistrer l'optimality gap pour chaque agent
            optimality_gap_agent = np.linalg.norm(alpha_agent - alpha_star_centralized)
            optimality_gap_by_agent_dgd[agent_id].append(optimality_gap_agent)
            
        agent_alphas = new_agent_alphas
        
        avg_alpha = np.mean(agent_alphas, axis=0)
        
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgd.append(optimality_gap)
        
        if verbose:
            print(f"DGD - Iteration {iteration+1}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")

    return agent_alphas, optimality_gap_history_dgd, optimality_gap_by_agent_dgd

def gradient_tracking(
    agents_data_indices,
    agents_x_data,
    agents_y_data,
    X_m_points,
    Kmm,
    communication_graph,
    step_size,
    num_iterations,
    alpha_star_centralized,
    verbose=False):
    """Implémentation du Gradient Tracking (GT)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    
    agents_Knm = [kernel_matrix(agents_x_data[i], X_m_points) for i in range(num_agents)]
    
    W = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        neighbors = list(communication_graph.neighbors(i))
        for j in neighbors:
            W[i, j] = communication_graph.get_edge_data(i, j)['weight']

    agent_gradients = []
    for agent_id in range(num_agents):
        Knm_agent = agents_Knm[agent_id]
        y_agent = agents_y_data[agent_id]
        alpha_agent = agent_alphas[agent_id]
        grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent
        agent_gradients.append(grad_local)
    
    optimality_gap_history_dgt = []
    # Suivi de l'optimality gap par agent
    optimality_gap_by_agent_dgt = [[] for _ in range(num_agents)]

    for iteration in range(num_iterations):
        new_agent_alphas = [np.zeros(m) for _ in range(num_agents)]
        new_agent_gradients = [np.zeros(m) for _ in range(num_agents)]
        for agent_id in range(num_agents):
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            alpha_avg = np.zeros(m)
            grad_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id))
            for neighbor_id in neighbors:
                alpha_avg += W[agent_id, neighbor_id] * agent_alphas[neighbor_id]
                grad_avg += W[agent_id, neighbor_id] * agent_gradients[neighbor_id]
            
            new_alpha_agent = alpha_avg - step_size * agent_gradients[agent_id]
            new_agent_alphas[agent_id] = new_alpha_agent
            
            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent
            new_grad_local = (sigma**2 / num_agents) * Kmm @ new_alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ new_alpha_agent)) + (nu / num_agents) * new_alpha_agent
            
            new_agent_gradients[agent_id] = grad_avg + (new_grad_local - grad_local)
            
            # Calculer et enregistrer l'optimality gap pour chaque agent
            optimality_gap_agent = np.linalg.norm(alpha_agent - alpha_star_centralized)
            optimality_gap_by_agent_dgt[agent_id].append(optimality_gap_agent)

        agent_alphas = new_agent_alphas
        agent_gradients = new_agent_gradients
        
        avg_alpha = np.mean(agent_alphas, axis=0)
        
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgt.append(optimality_gap)
        
        if verbose:
            print(f"DGT - Iteration {iteration+1}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")

    return agent_alphas, optimality_gap_history_dgt, optimality_gap_by_agent_dgt

def dual_decomposition(
    agents_data_indices,
    agents_x_data,
    agents_y_data,
    X_m_points,
    Kmm, 
    communication_graph,
    step_size,
    num_iterations,
    alpha_star_centralized,
    verbose=False):
    """Implémentation de la Décomposition Duale (DD) pair à pair."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    
    lambdas = {}
    for i in range(num_agents):
        for j in list(communication_graph.neighbors(i)):
            if j < i:
                lambdas[(i,j)] = np.zeros(m)
            if j > i:
                lambdas[(j,i)] = np.zeros(m)
    
    agents_Knm = [kernel_matrix(agents_x_data[i], X_m_points) for i in range(num_agents)]
    
    agents_H = []
    agents_b = []
    for agent_id in range(num_agents):
        Knm_agent = agents_Knm[agent_id]
        y_agent = agents_y_data[agent_id]
        
        # Calcul de H_i
        H_agent = (sigma**2 / num_agents) * Kmm + Knm_agent.T @ Knm_agent + (nu / num_agents) * np.eye(m)
        agents_H.append(H_agent)
        agents_b.append(Knm_agent.T @ y_agent)
    
    optimality_gap_history_dd = []
    # Suivi de l'optimality gap par agent
    optimality_gap_by_agent_dd = [[] for _ in range(num_agents)]
    
    for iteration in range(num_iterations):
        for agent_id in range(num_agents):
            dual_term = np.zeros(m)
            for neighbor_id in list(communication_graph.neighbors(agent_id)):
                if neighbor_id == agent_id:
                    continue
                if neighbor_id < agent_id:
                    dual_term += lambdas[(agent_id, neighbor_id)]
                else:
                    dual_term -= lambdas[(neighbor_id, agent_id)]
            
            agent_alphas[agent_id] = np.linalg.solve(agents_H[agent_id], agents_b[agent_id] - dual_term)
            
            # Calculer et enregistrer l'optimality gap pour chaque agent
            optimality_gap_agent = np.linalg.norm(agent_alphas[agent_id] - alpha_star_centralized)
            optimality_gap_by_agent_dd[agent_id].append(optimality_gap_agent)
        
        for i in range(num_agents):
            for j in list(communication_graph.neighbors(i)):
                if j < i:
                    lambdas[(i,j)] = lambdas[(i,j)] + step_size * (agent_alphas[i] - agent_alphas[j])
        
        consensus_error = 0
        for i in range(num_agents):
            for j in list(communication_graph.neighbors(i)):
                if j < i:
                    consensus_error += np.linalg.norm(agent_alphas[i] - agent_alphas[j])
        
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dd.append(optimality_gap)
        
        if verbose:
            print(f"DD - Iteration {iteration}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")
    
    return agent_alphas, optimality_gap_history_dd, optimality_gap_by_agent_dd

def admm(
    agents_data_indices,
    agents_x_data,
    agents_y_data,
    X_m_points,
    Kmm,
    communication_graph,
    beta,
    num_iterations,
    alpha_star_centralized,
    verbose=False):
    """Implémentation de l'ADMM."""
    
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    
    y = {}
    lambdas = {}
    for agent in range(num_agents):
        for neighbor in list(communication_graph.neighbors(agent)):
            if neighbor == agent:
                continue
            if neighbor < agent:
                y[(agent, neighbor)] = np.zeros(m)
                lambdas[(agent, neighbor)] = np.zeros(m)
            if neighbor > agent:
                y[(neighbor, agent)] = np.zeros(m)
                lambdas[(neighbor, agent)] = np.zeros(m)
    
    agents_Knm = [kernel_matrix(agents_x_data[i], X_m_points) for i in range(num_agents)]
    
    agents_H = []
    agents_b = []
    for agent_id in range(num_agents):
        Knm_agent = agents_Knm[agent_id]
        y_agent = agents_y_data[agent_id]
        
        # Calcul de H_i
        H_agent = (sigma**2 / num_agents) * Kmm + Knm_agent.T @ Knm_agent + (nu / num_agents) * np.eye(m)
        agents_H.append(H_agent)
        agents_b.append(Knm_agent.T @ y_agent)
    
    optimality_gap_history_admm = []
    # Suivi de l'optimality gap par agent
    optimality_gap_by_agent_admm = [[] for _ in range(num_agents)]
    
    for iteration in range(num_iterations):
        for agent in range(num_agents):
            
            neighbors = communication_graph.neighbors(agent)
            d_agent = sum(1 for neighbor in neighbors if neighbor != agent)
            
            dual_term = np.zeros(m)
            for neighbor in neighbors:
                if neighbor == agent:
                    continue
                key = tuple(sorted((agent, neighbor)))
                
                if agent < neighbor:
                    dual_term += beta * y[key] - lambdas[key]
                else:
                    dual_term += beta * y[key] + lambdas[key]

            agent_alphas[agent] = np.linalg.solve(agents_H[agent] + beta * d_agent * np.eye(m), agents_b[agent] + dual_term)

            # Tracking du gap local
            optimality_gap_agent = np.linalg.norm(agent_alphas[agent] - alpha_star_centralized)
            optimality_gap_by_agent_admm[agent].append(optimality_gap_agent)
        
        # Mise à jour de y et lambda
        for agent in range(num_agents):
            for neighbor in list(communication_graph.neighbors(agent)):
                if neighbor == agent:
                    continue
                if neighbor < agent:
                    y[(agent, neighbor)] = 0.5 * (agent_alphas[agent] + agent_alphas[neighbor])
                    
                if neighbor > agent:
                    y[(neighbor, agent)] = 0.5 * (agent_alphas[agent] + agent_alphas[neighbor])

        for agent in range(num_agents):
            for neighbor in list(communication_graph.neighbors(agent)):
                if neighbor == agent:
                    continue
                if neighbor < agent:
                    lambdas[(agent, neighbor)] += beta * (agent_alphas[agent] - y[(agent, neighbor)])
                if neighbor > agent:
                    lambdas[(neighbor, agent)] += beta * (agent_alphas[neighbor] - y[(neighbor, agent)])

        # Optimality gap global
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_admm.append(optimality_gap)
        
        if verbose:
            print(f"ADMM - Iteration {iteration}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")
        
    return agent_alphas, optimality_gap_history_admm, optimality_gap_by_agent_admm

# --- Algorithme de la partie 2 ---
def federated_averaging(
    agents_X,
    agents_Y,
    X_m_points,
    Kmm,
    num_rounds,
    epochs_per_round,
    batch_size,
    learning_rate,
    client_selection_prob=1.0,
    use_decreasing_lr=False):
    """Implémentation de Federated Averaging (FedAvg)."""
    num_agents = len(agents_X)
    m = len(X_m_points)
    global_alpha = np.zeros(m)

    objective_error_history_fedavg = []
    client_sample_counts = np.array([len(agent_x) for agent_x in agents_X])
    agents_Knm_full = [kernel_matrix(agents_X[i], X_m_points) for i in range(num_agents)]

    for round_num in range(num_rounds):
        selected_clients = []
        for agent_id in range(num_agents):
            if np.random.random() < client_selection_prob:
                selected_clients.append(agent_id)
        
        if not selected_clients:
            selected_clients = [np.random.randint(0, num_agents)]
            
        #print(f"Round {round_num+1}/{num_rounds}, {len(selected_clients)}/{num_agents} clients sélectionnés")
        
        local_alphas = []
        sample_counts_selected = []
        
        for agent_id in selected_clients:
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent_full = agents_Knm_full[agent_id]
            local_alpha = global_alpha.copy()
            
            for epoch in range(epochs_per_round):
                
                num_samples = len(agent_x)
                num_batches = max(1, num_samples // batch_size)
                
                indices_all = np.random.permutation(num_samples)
                
                for batch_idx in range(num_batches):
                    
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    if start_idx >= end_idx:
                        continue
                        
                    indices_batch = indices_all[start_idx:end_idx]
                    
                    x_batch = [agent_x[i] for i in indices_batch]
                    y_batch = [agent_y[i] for i in indices_batch]
                    
                    Knm_batch = kernel_matrix(x_batch, X_m_points)
                    
                    grad_local = (sigma**2 / num_agents) * Kmm @ local_alpha - \
                                (Knm_batch.T @ (y_batch - Knm_batch @ local_alpha)) + \
                                (nu / num_agents) * local_alpha
                    
                    current_lr = learning_rate
                    if use_decreasing_lr:
                        global_iter = round_num * epochs_per_round * num_batches + epoch * num_batches + batch_idx
                        current_lr = learning_rate / (1 + 0.01 * global_iter)
                    
                    local_alpha = local_alpha - current_lr * grad_local
            
            local_alphas.append(local_alpha)
            sample_counts_selected.append(client_sample_counts[agent_id])
        
        sample_counts_selected = np.array(sample_counts_selected)
        weights = sample_counts_selected / np.sum(sample_counts_selected)
        
        global_alpha = np.zeros(m)
        for i, alpha in enumerate(local_alphas):
            global_alpha += weights[i] * alpha

        objective_error = 0
        for agent_id in range(num_agents):
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent = agents_Knm_full[agent_id]
            
            reg_term = (sigma**2 / (2*num_agents)) * global_alpha.T @ Kmm @ global_alpha + \
                       (nu / (2*num_agents)) * np.linalg.norm(global_alpha)**2
            
            error_term = (1/(2*num_agents)) * np.sum((agent_y - Knm_agent @ global_alpha)**2)
            
            objective_error += reg_term + error_term
            
        objective_error_history_fedavg.append(objective_error)
        #print(f"Round {round_num+1}/{num_rounds}, Objective Error: {objective_error:.4f}")

    return global_alpha, objective_error_history_fedavg

# --- Algorithme de la partie 3 ---
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
    noise_std):
    """Implémentation de DGD avec Privacité Différentielle (DGD-DP)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    optimality_gap_history_dgd_dp = []

    for _ in range(num_iterations):
        for agent_id in range(num_agents):
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent

            noise = np.random.normal(0, noise_std, size=grad_local.shape)
            grad_local_noisy = grad_local + noise

            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id)) + [agent_id]
            for neighbor_id in neighbors:
                alpha_avg += agent_alphas[neighbor_id]
            alpha_avg /= len(neighbors)

            agent_alphas[agent_id] = alpha_avg - step_size * grad_local_noisy

        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgd_dp.append(optimality_gap)

    return agent_alphas, optimality_gap_history_dgd_dp

# --- Partie principale du script ---
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
        #visualize_graph(graph, graph_name)
        
        print("--- 1.1 Decentralized Gradient Descent (DGD) ---")
        step_size_dgd = 0.002
        agent_alphas_dgd, optimality_gap_history_dgd, optimality_gap_by_agent_dgd = decentralized_gradient_descent(
            agents_data_indices,
            agents_x_data,
            agents_y_data,
            X_m_points,
            Kmm,
            communication_graph,
            step_size_dgd,
            num_iterations,
            alpha_star_centralized,
            verbose=False
        )
        iterations_dgd = range(num_iterations)
        # Afficher la convergence par agent pour DGD
        #plot_convergence_by_agent(iterations_dgd, optimality_gap_by_agent_dgd, "DGD", graph, step_size_dgd)
        avg_alpha_dgd = np.mean(agent_alphas_dgd, axis=0)

        print("--- 1.2 Gradient Tracking (GT) ---")
        step_size_gt = 0.002
        agent_alphas_gt, optimality_gap_history_gt, optimality_gap_by_agent_gt = gradient_tracking(
            agents_data_indices,
            agents_x_data,
            agents_y_data,
            X_m_points,
            Kmm,
            communication_graph,
            step_size_gt,
            num_iterations,
            alpha_star_centralized,
            verbose=False
        )
        iterations_gt = range(num_iterations)
        # Afficher la convergence par agent pour GT
        #plot_convergence_by_agent(iterations_gt, optimality_gap_by_agent_gt, "GradientTracking", graph, step_size_gt)
        avg_alpha_gt = np.mean(agent_alphas_gt, axis=0)

        print("--- 1.3 Dual Decomposition (DD) ---")
        step_size_dd = 0.1
        agent_alphas_dd, optimality_gap_history_dd, optimality_gap_by_agent_dd = dual_decomposition(
            agents_data_indices,
            agents_x_data,
            agents_y_data,
            X_m_points,
            Kmm,
            communication_graph,
            step_size_dd,
            num_iterations,
            alpha_star_centralized,
            verbose=False
        )
        iterations_dd = range(num_iterations)
        # Afficher la convergence par agent pour DD
        #plot_convergence_by_agent(iterations_dd, optimality_gap_by_agent_dd, "DualDecomposition", graph, step_size_dd)
        avg_alpha_dd = np.mean(agent_alphas_dd, axis=0)

        print("--- 1.4 ADMM ---")
        beta_admm = 0.01
        agent_alphas_admm, optimality_gap_history_admm, optimality_gap_by_agent_admm = admm(
            agents_data_indices,
            agents_x_data,
            agents_y_data,
            X_m_points,
            Kmm,
            graph,
            beta_admm,
            num_iterations,
            alpha_star_centralized,
            verbose=False
        )
        iterations_admm = range(len(optimality_gap_history_admm))
        # Afficher la convergence par agent pour ADMM
        plot_convergence_by_agent(iterations_admm, optimality_gap_by_agent_admm, "ADMM", graph, beta_admm)
        avg_alpha_admm = np.mean(agent_alphas_admm, axis=0)
        
        # Afficher la comparaison des courbes de convergence par agent pour toutes les méthodes
        min_length = min(
            len(iterations_dgd),
            len(iterations_gt),
            len(iterations_dd),
            len(iterations_admm)
        )
        
        # Tronquer les données à la longueur minimale
        methods_agent_convergence = {
            "DGD": [gap_hist[:min_length] for gap_hist in optimality_gap_by_agent_dgd],
            "GradientTracking": [gap_hist[:min_length] for gap_hist in optimality_gap_by_agent_gt],
            "DualDecomposition": [gap_hist[:min_length] for gap_hist in optimality_gap_by_agent_dd],
            "ADMM": [gap_hist[:min_length] for gap_hist in optimality_gap_by_agent_admm]
        }
        
        plot_convergence_agents_multi(range(min_length), methods_agent_convergence)
        
        print(f"Lengths - DGD: {len(optimality_gap_history_dgd)}, GT: {len(optimality_gap_history_gt)}, DD: {len(optimality_gap_history_dd)}, ADMM: {len(optimality_gap_history_admm)}")
        
        min_length = min(len(optimality_gap_history_dgd), 
                        len(optimality_gap_history_gt),
                        len(optimality_gap_history_dd), 
                        len(optimality_gap_history_admm))
                        
        convergence_data = {
            "DGD": optimality_gap_history_dgd[:min_length],
            "GradientTracking": optimality_gap_history_gt[:min_length],
            "DualDecomposition": optimality_gap_history_dd[:min_length],
            "ADMM": optimality_gap_history_admm[:min_length]
        }
        iterations = range(min_length)
        plot_convergence_multi(iterations, convergence_data)

        print("Visualisation comparée de toutes les méthodes")
        methods_alpha = {
            "Centralized": alpha_star_centralized,
            "DGD": avg_alpha_dgd,
            "GradientTracking": avg_alpha_gt,
            "DualDecomposition": avg_alpha_dd,
            "ADMM": avg_alpha_admm
        }
        plot_reconstruction_multi(x_prime_grid, methods_alpha, X_m_points, y_nystrom, (x_data[:n_total], y_data[:n_total]))
    
    print("--- 2. Federated Averaging (FedAvg) ---")
    data_part2 = load_data('data/second_database.pkl')
    
    # Debug information to understand data structure
    print(f"Type of data_part2: {type(data_part2)}")
    print(f"Structure of data_part2: {len(data_part2)} elements")
    
    # Check if it's a list with two elements like the first dataset
    if isinstance(data_part2, list) and len(data_part2) == 2:
        agents_X_part2, agents_Y_part2 = data_part2[0], data_part2[1]
        print(f"Loaded data as list with {len(agents_X_part2)} agents for X and {len(agents_Y_part2)} agents for Y")
    # Check if it's already a list of agents directly
    elif isinstance(data_part2, list):
        # If the data is a list of agents, with each agent having X and Y data
        if len(data_part2) > 0 and isinstance(data_part2[0], (list, tuple)) and len(data_part2[0]) == 2:
            agents_X_part2 = [agent[0] for agent in data_part2]
            agents_Y_part2 = [agent[1] for agent in data_part2]
            print(f"Loaded data as list of {len(agents_X_part2)} agent tuples")
        # Last resort - assume it's a flat list that needs to be divided
        else:
            # Assume we need the same number of agents as part 1
            num_agents = 5  # Same as part 1
            # Divide the data into equal parts for each agent
            data_size = len(data_part2)
            chunk_size = data_size // num_agents
            agents_X_part2 = [data_part2[i:i+chunk_size] for i in range(0, data_size, chunk_size)]
            # Just reuse X data as Y data for now (this is a placeholder)
            agents_Y_part2 = agents_X_part2
            print(f"Warning: Unclear data structure, divided into {len(agents_X_part2)} chunks")
    else:
        raise TypeError("Unexpected structure for data_part2. Please check the data format.")

    n_total_part2 = 100 # Total points (même si déjà divisées, pour m_nystrom)
    m_nystrom_part2 = 10
    X_m_points_part2 = np.linspace(-1, 1, m_nystrom_part2) # Points Nyström pour la Partie 2
    Kmm_part2 = kernel_matrix(X_m_points_part2, X_m_points_part2) # Matrice Kmm pour la Partie 2

    num_rounds_fedavg = 50
    epochs_per_round_fedavg = 1
    batch_size_fedavg = 20
    learning_rate_fedavg = 0.01

    global_alpha_fedavg, objective_error_history_fedavg = federated_averaging(
        agents_X_part2, agents_Y_part2, X_m_points_part2, Kmm_part2,
        num_rounds_fedavg, epochs_per_round_fedavg, batch_size_fedavg, learning_rate_fedavg,
        client_selection_prob=0.8, use_decreasing_lr=True  # Add these new parameters
    )

    iterations_fedavg = range(num_rounds_fedavg)
    plot_convergence(iterations_fedavg, objective_error_history_fedavg, "FedAvg_ObjectiveError") # Convergence erreur objective
    x_prime_grid_fedavg = np.linspace(-1, 1, 250)
    visualize_function(x_prime_grid_fedavg, global_alpha_fedavg, X_m_points_part2, "FedAvg",  y_nystrom, (x_data[:n_total], y_data[:n_total]))



    print("--- 3. DGD-DP ---")
    communication_graph_dp = communication_graph.copy() # Réutiliser le graphe de DGD
    step_size_dgd_dp = 0.01
    num_iterations_dgd_dp = 1000
    noise_std_dp = 0.1 # Écart-type du bruit gaussien - à calibrer pour epsilon

    agent_alphas_dgd_dp, optimality_gap_history_dgd_dp = dgd_dp(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dp,
        step_size_dgd_dp, num_iterations_dgd_dp, alpha_star_centralized, noise_std_dp
    )

    iterations_dgd_dp = range(num_iterations_dgd_dp)
    plot_convergence(iterations_dgd_dp, optimality_gap_history_dgd_dp, "DGD_DP")
    avg_alpha_dgd_dp = np.mean(agent_alphas_dgd_dp, axis=0)
    visualize_function(x_prime_grid, avg_alpha_dgd_dp, X_m_points, "DGD_DP",  y_nystrom, (x_data[:n_total], y_data[:n_total]))

    print("--- Programme terminé ---")