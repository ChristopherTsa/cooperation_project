import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx  # Pour les graphes de communication

# --- Paramètres Globaux ---
sigma = 0.5
nu = 1.0

# --- Fonctions Utilitaires ---
def load_data(filename):
    """Charge les données à partir d'un fichier pickle."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

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

def visualize_function(x_prime, alpha, X_m_points, algorithm_name):
    """Visualise la fonction apprise sur une grille uniforme."""
    nt = len(x_prime)
    f_x_prime = np.zeros(nt)
    for i in range(nt):
        for j in range(len(X_m_points)):
            f_x_prime[i] += alpha[j] * euclidean_kernel(x_prime[i], X_m_points[j])

    plt.figure()
    plt.plot(x_prime, f_x_prime)
    plt.scatter(X_m_points, np.zeros_like(X_m_points), color='red', marker='x', label='Points Nyström')
    plt.title(f"Fonction apprise avec {algorithm_name}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/learned_function/learned_fonction_{algorithm_name}.pdf")
    plt.show()

# --- Fonctions pour les Algorithmes Distribués ---

def decentralized_gradient_descent(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, step_size, num_iterations, alpha_star_centralized):
    """Implémentation de la Descente de Gradient Décentralisée (DGD)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    optimality_gap_history_dgd = []

    for iteration in range(num_iterations):
        for agent_id in range(num_agents):
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent

            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id)) + [agent_id]
            for neighbor_id in neighbors:
                alpha_avg += agent_alphas[neighbor_id]
            alpha_avg /= len(neighbors)

            agent_alphas[agent_id] = alpha_avg - step_size * grad_local

        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgd.append(optimality_gap)
        #print(f"DGD - Iteration {iteration+1}/{num_iterations}, Optimality Gap: {optimality_gap:.6f}")

    return agent_alphas, optimality_gap_history_dgd


def gradient_tracking(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, step_size, num_iterations, alpha_star_centralized):
    """Implémentation de Gradient Tracking (GT)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agent_gradients = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    for agent_id in range(num_agents):
        Knm_agent = agents_Knm[agent_id]
        y_agent = agents_y_data[agent_id]
        alpha_agent = agent_alphas[agent_id]
        agent_gradients[agent_id] = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent

    optimality_gap_history_gt = []

    for iteration in range(num_iterations):
        new_agent_alphas = [np.zeros(m) for _ in range(num_agents)]
        new_agent_gradients = [np.zeros(m) for _ in range(num_agents)]

        for agent_id in range(num_agents):
            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id)) + [agent_id]
            for neighbor_id in neighbors:
                alpha_avg += agent_alphas[neighbor_id]
            alpha_avg /= len(neighbors)

            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            current_grad_local = (sigma**2 / num_agents) * Kmm @ agent_alphas[agent_id] - (Knm_agent.T @ (y_agent - Knm_agent @ agent_alphas[agent_id])) + (nu / num_agents) * agent_alphas[agent_id]

            gradient_avg = np.zeros(m)
            for neighbor_id in neighbors:
                gradient_avg += agent_gradients[neighbor_id]
            gradient_avg /= len(neighbors)

            new_agent_alphas[agent_id] = alpha_avg - step_size * agent_gradients[agent_id]
            new_agent_gradients[agent_id] = gradient_avg + current_grad_local - agent_gradients[agent_id]

        agent_alphas = new_agent_alphas
        agent_gradients = new_agent_gradients

        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_gt.append(optimality_gap)

    return agent_alphas, optimality_gap_history_gt


def dual_decomposition(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, step_size_primal, step_size_dual, rho, num_iterations, alpha_star_centralized):
    """Implémentation de la Décomposition Duale (DD)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)] # Variables primales locales
    agent_lambdas = [np.zeros(m) for _ in range(num_agents)] # Variables duales locales
    z_global = np.zeros(m) # Variable primale globale (consensus)
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    optimality_gap_history_dd = []

    for iteration in range(num_iterations):
        z_avg = np.zeros(m)
        for agent_id in range(num_agents):
            # 1. Mise à jour de alpha (primale locale)
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            lambda_agent = agent_lambdas[agent_id]
            alpha_agent_prev = agent_alphas[agent_id] # Pour la mise à jour de lambda ##inutil ?

            # Gradient de la fonction augmentée de Lagrange (par rapport à alpha_agent)
            grad_L_alpha = (sigma**2 / num_agents) * Kmm @ agent_alphas[agent_id] - (Knm_agent.T @ (y_agent - Knm_agent @ agent_alphas[agent_id])) + (nu / num_agents) * agent_alphas[agent_id] - lambda_agent + rho * (agent_alphas[agent_id] - z_global)

            agent_alphas[agent_id] = agent_alphas[agent_id] - step_size_primal * grad_L_alpha # Mise à jour primale (gradient ascent sur -L)

            # Contribution à la moyenne de z pour la mise à jour duale (calculée après la mise à jour primale de tous les agents)
            z_avg += agent_alphas[agent_id]


        # 2. Mise à jour de z (primale globale - consensus)
        z_global = z_avg / num_agents # Moyenne des alphas locaux

        # 3. Mise à jour de lambda (duale) - Pour chaque agent
        for agent_id in range(num_agents):
            agent_lambdas[agent_id] = agent_lambdas[agent_id] + step_size_dual * rho * (agent_alphas[agent_id] - z_global) # Mise à jour duale (gradient ascent sur L par rapport à lambda)


        # Calcul de l'optimality gap (comparaison de z_global avec alpha_star_centralized)
        optimality_gap = np.linalg.norm(z_global - alpha_star_centralized)
        optimality_gap_history_dd.append(optimality_gap)

    return agent_alphas, optimality_gap_history_dd, z_global



def admm(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, rho, num_iterations, alpha_star_centralized):
    """Implémentation de ADMM."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)] # Variables primales locales
    agent_lambdas = [np.zeros(m) for _ in range(num_agents)] # Variables duales locales
    z_global = np.zeros(m) # Variable primale globale (consensus) - initialisée à zéro
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    optimality_gap_history_admm = []

    for iteration in range(num_iterations):
        z_avg = np.zeros(m) # Pour accumuler la somme des alphas pour la mise à jour de z

        for agent_id in range(num_agents):
            # 1. Mise à jour de alpha (étape de minimisation locale - mise à jour primale)
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            lambda_agent = agent_lambdas[agent_id]

            # Solution analytique pour alpha_agent (dérivée de la fonction augmentée de Lagrange = 0)
            # Formule dérivée en posant la dérivée par rapport à alpha_agent de la fonction augmentée de Lagrange égale à zéro
            agent_alphas[agent_id] = np.linalg.solve((sigma**2 / num_agents) * Kmm + Knm_agent.T @ Knm_agent + (nu / num_agents) * np.eye(m) + rho * np.eye(m),
                                             Knm_agent.T @ y_agent + lambda_agent + rho * z_global) # Mise à jour de alpha_agent

            # Contribution à la moyenne de z pour la mise à jour duale (calculée après la mise à jour primale de tous les agents)
            z_avg += agent_alphas[agent_id]


        # 2. Mise à jour de z (étape de consensus - mise à jour primale globale)
        z_global = z_avg / num_agents # Mise à jour de z_global (consensus sur les alphas locaux)

        # 3. Mise à jour de lambda (étape duale) - Pour chaque agent
        for agent_id in range(num_agents):
            agent_lambdas[agent_id] = agent_lambdas[agent_id] + rho * (agent_alphas[agent_id] - z_global) # Mise à jour de lambda_agent (variable duale)


        # Calcul de l'optimality gap (comparaison de z_global avec alpha_star_centralized)
        optimality_gap = np.linalg.norm(z_global - alpha_star_centralized)
        optimality_gap_history_admm.append(optimality_gap)

    return agent_alphas, optimality_gap_history_admm, z_global


def federated_averaging(agents_X, agents_Y, X_m_points, Kmm, num_rounds, epochs_per_round, batch_size, learning_rate):
    """Implémentation de Federated Averaging (FedAvg)."""
    num_agents = len(agents_X)
    m = len(X_m_points)
    global_alpha = np.zeros(m)

    objective_error_history_fedavg = []

    for round_num in range(num_rounds):
        selected_agent_indices = list(range(num_agents)) # Tous les agents sont sélectionnés

        local_alphas = []
        for agent_id in selected_agent_indices:
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent = kernel_matrix(agent_x, X_m_points)
            local_alpha = global_alpha.copy()

            for epoch in range(epochs_per_round):
                indices_batch = np.random.choice(len(agent_x), batch_size, replace=False)
                x_batch = [agent_x[i] for i in indices_batch]
                y_batch = [agent_y[i] for i in indices_batch]
                Knm_batch = kernel_matrix(x_batch, X_m_points)

                grad_local = (sigma**2 / num_agents) * Kmm @ local_alpha - (Knm_batch.T @ (y_batch - Knm_batch @ local_alpha)) + (nu / num_agents) * local_alpha
                local_alpha = local_alpha - learning_rate * grad_local

            local_alphas.append(local_alpha)

        global_alpha = np.mean(local_alphas, axis=0)

        # Calcul de l'erreur objective globale (pour FedAvg)
        objective_error = 0
        for agent_id in range(num_agents): # Boucle sur tous les agents pour calculer l'erreur globale
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent = kernel_matrix(agent_x, X_m_points)
            objective_error += (sigma**2 / (2*num_agents)) * global_alpha.T @ Kmm @ global_alpha + (1/(2*num_agents)) * np.sum((agent_y - Knm_agent @ global_alpha)**2) + (nu / (2*num_agents)) * np.linalg.norm(global_alpha)**2 # Somme des fonctions objectives locales évaluées au modèle global
        objective_error_history_fedavg.append(objective_error)


        print(f"Round {round_num+1}/{num_rounds}, Objective Error: {objective_error:.4f}")

    return global_alpha, objective_error_history_fedavg



def dgd_dp(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, step_size, num_iterations, alpha_star_centralized, noise_std):
    """Implémentation de DGD avec Privacité Différentielle (DGD-DP)."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    optimality_gap_history_dgd_dp = []

    for iteration in range(num_iterations):
        for agent_id in range(num_agents):
            # 1. Calcul du gradient local
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            alpha_agent = agent_alphas[agent_id]

            grad_local = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent

            # 2. Ajout de bruit pour la privacité différentielle (bruit gaussien)
            noise = np.random.normal(0, noise_std, size=grad_local.shape) # Bruit gaussien
            grad_local_noisy = grad_local + noise # Gradient bruité

            # 3. Communication et Agrégation des alphas voisins (en utilisant le gradient bruité)
            alpha_avg = np.zeros(m)
            neighbors = list(communication_graph.neighbors(agent_id)) + [agent_id]
            for neighbor_id in neighbors:
                alpha_avg += agent_alphas[neighbor_id]
            alpha_avg /= len(neighbors)

            # 4. Mise à jour de alpha avec le gradient bruité
            agent_alphas[agent_id] = alpha_avg - step_size * grad_local_noisy # Utilisation du gradient bruité pour la mise à jour

        # Calcul de l'optimality gap
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dgd_dp.append(optimality_gap)

    return agent_alphas, optimality_gap_history_dgd_dp


# --- Fonction pour résoudre le problème centralisé ---
def centralized_solution(Kmm, Knm, y, sigma, nu):
    """Calcule la solution centralisée pour comparer les méthodes distribuées."""
    m = Kmm.shape[0]
    alpha_star = np.linalg.solve(sigma**2 * Kmm + Knm.T @ Knm + nu * np.eye(m), Knm.T @ y)
    return alpha_star


# --- Partie Principale du Script ---
if __name__ == "__main__":
    print("--- Partie 1: Méthodes Distribuées Classiques (DGD, GT, DD, ADMM) ---")

    # 1. Charger les données
    data_part1 = load_data('data/first_database.pkl')
    x_data, y_data = data_part1[0], data_part1[1]

    # 2. Choisir n, m, a et diviser les données
    n_total = 100
    m_nystrom = 10
    num_agents = 5
    points_per_agent = n_total // num_agents

    X_m_points, M_indices = nystrom_approximation(x_data[:n_total], m_nystrom)

    agents_data_indices = [list(range(i*points_per_agent, (i+1)*points_per_agent)) for i in range(num_agents)]
    agents_x_data = [[x_data[i] for i in indices] for indices in agents_data_indices]
    agents_y_data = [[y_data[i] for i in indices] for indices in agents_data_indices]

    Kmm = kernel_matrix(X_m_points, X_m_points)

    # --- Calcul de la solution centralisée ---
    print("Calcul de la solution centralisée...")
    Knm_centralized = kernel_matrix(x_data[:n_total], X_m_points)
    alpha_star_centralized = centralized_solution(Kmm, Knm_centralized, y_data[:n_total], sigma, nu)
    print("Solution centralisée calculée.")


    # --- 1. Decentralized Gradient Descent (DGD) ---
    print("--- Decentralized Gradient Descent (DGD) ---")
    communication_graph_dgd = nx.Graph()
    communication_graph_dgd.add_nodes_from(range(num_agents))
    communication_graph_dgd.add_edges_from([(i, (i+1)%num_agents) for i in range(num_agents)]) # Graphe en anneau

    step_size_dgd = 0.01
    num_iterations_dgd = 1000

    agent_alphas_dgd, optimality_gap_history_dgd = decentralized_gradient_descent(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dgd,
        step_size_dgd, num_iterations_dgd, alpha_star_centralized
    )

    iterations_dgd = range(num_iterations_dgd)
    plot_convergence(iterations_dgd, optimality_gap_history_dgd, "DGD")
    avg_alpha_dgd = np.mean(agent_alphas_dgd, axis=0)
    x_prime_grid = np.linspace(-1, 1, 250)
    visualize_function(x_prime_grid, avg_alpha_dgd, X_m_points, "DGD")


    # --- 2. Gradient Tracking (GT) ---
    print("--- Gradient Tracking (GT) ---")
    communication_graph_gt = communication_graph_dgd.copy()
    step_size_gt = 0.015 # Ajuster step size pour GT
    agent_alphas_gt, optimality_gap_history_gt = gradient_tracking(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_gt,
        step_size_gt, num_iterations_dgd, alpha_star_centralized # Réutiliser num_iterations_dgd
    )

    iterations_gt = range(num_iterations_dgd)
    plot_convergence(iterations_gt, optimality_gap_history_gt, "GradientTracking")
    avg_alpha_gt = np.mean(agent_alphas_gt, axis=0)
    visualize_function(x_prime_grid, avg_alpha_gt, X_m_points, "GradientTracking")


    # --- 3. Dual Decomposition (DD) ---
    print("--- Dual Decomposition (DD) ---")
    communication_graph_dd = communication_graph_dgd.copy()
    step_size_primal_dd = 0.01
    step_size_dual_dd = 0.01
    rho_dd = 0.1 # Paramètre de pénalité de l'ADMM, à ajuster
    agent_alphas_dd, optimality_gap_history_dd, z_global_dd = dual_decomposition(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dd,
        step_size_primal_dd, step_size_dual_dd, rho_dd, num_iterations_dgd, alpha_star_centralized
    )

    iterations_dd = range(num_iterations_dgd)
    plot_convergence(iterations_dd, optimality_gap_history_dd, "DualDecomposition")
    visualize_function(x_prime_grid, z_global_dd, X_m_points, "DualDecomposition") # Utiliser z_global pour visualiser


    # --- 4. ADMM ---
    print("--- ADMM ---")
    communication_graph_admm = communication_graph_dgd.copy()
    rho_admm = 0.1 # Paramètre de pénalité de l'ADMM, à ajuster
    agent_alphas_admm, optimality_gap_history_admm, z_global_admm = admm(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, rho_admm,
        num_iterations_dgd, alpha_star_centralized # Réutiliser num_iterations_dgd
    )

    iterations_admm = range(num_iterations_dgd)
    plot_convergence(iterations_admm, optimality_gap_history_admm, "ADMM")
    visualize_function(x_prime_grid, z_global_admm, X_m_points, "ADMM") # Utiliser z_global pour visualiser

""" 
    # --- Partie 2: Federated Averaging (FedAvg) ---
    print("--- Partie 2: Federated Averaging (FedAvg) ---")
    data_part2 = load_data('data/second_database.pkl')
    agents_X_part2, agents_Y_part2 = data_part2['X'], data_part2['Y'] # Données déjà divisées par agent

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
        num_rounds_fedavg, epochs_per_round_fedavg, batch_size_fedavg, learning_rate_fedavg
    )

    iterations_fedavg = range(num_rounds_fedavg)
    plot_convergence(iterations_fedavg, objective_error_history_fedavg, "FedAvg_ObjectiveError") # Convergence erreur objective
    x_prime_grid_fedavg = np.linspace(-1, 1, 250)
    visualize_function(x_prime_grid_fedavg, global_alpha_fedavg, X_m_points_part2, "FedAvg")


    # --- Partie 3: DGD-DP ---
    print("--- Partie 3: DGD-DP ---")
    communication_graph_dp = communication_graph_dgd.copy() # Réutiliser le graphe de DGD
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
    visualize_function(x_prime_grid, avg_alpha_dgd_dp, X_m_points, "DGD_DP")
 """

    #print("--- Fin du script principal ---")