import math as m
import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx  # Pour les graphes de communication
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
    plt.scatter(x_data, y_data, label='Données', s=10) # 's=10' pour ajuster la taille des points
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
    """Visualise la fonction apprise sur une grille uniforme et affiche
    les vraies valeurs y pour les points Nyström.
    Si selected_data est fourni sous la forme (x_selected, y_selected),
    on affiche également ces points."""
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
    """
    Affiche sur un même graphe la reconstruction des fonctions apprises
    pour chaque méthode.
    
    :param x_prime: Grille d'abscisses pour calculer la fonction.
    :param methods_alpha: Dictionnaire {nom_méthode: alpha} pour chaque méthode.
    :param X_m_points: Points Nyström.
    :param y_nystrom: Vraies valeurs y aux points Nyström.
    :param selected_data: Tuple (x_selected, y_selected) si besoin d'afficher les données complètes.
    """
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
    plt.title("Reconstruction Comparée des Méthodes")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/learned_function/comparison_reconstruction.pdf")
    plt.show()

def plot_convergence_multi(iterations, convergence_data):
    """
    Affiche sur un même graphe les courbes de convergence (optimality gap)
    pour chaque méthode itérative.
    
    :param iterations: Plage d'itérations (généralement range(num_iterations)).
    :param convergence_data: Dictionnaire {nom_méthode: liste_gap} pour chaque méthode.
    """
    plt.figure()
    for method_name, gap in convergence_data.items():
        plt.loglog(iterations, gap, label=f"{method_name}")
    plt.xlabel("Itérations")
    plt.ylabel("Optimality Gap")
    plt.title("Convergence Comparée des Méthodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/convergence/comparison_convergence.pdf")
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
    """Implémentation améliorée de Gradient Tracking (GT) avec stabilité accrue."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agent_gradients = [np.zeros(m) for _ in range(num_agents)]
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))
    
    # Création explicite de la matrice de poids W doublement stochastique
    W = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        neighbors = list(communication_graph.neighbors(i)) + [i]  # Add self-loop
        for j in neighbors:
            W[i, j] = 1.0 / len(neighbors)  # Equal weights
    
    # Initialisation des gradients locaux
    for agent_id in range(num_agents):
        Knm_agent = agents_Knm[agent_id]
        y_agent = agents_y_data[agent_id]
        alpha_agent = agent_alphas[agent_id]
        agent_gradients[agent_id] = (sigma**2 / num_agents) * Kmm @ alpha_agent - (Knm_agent.T @ (y_agent - Knm_agent @ alpha_agent)) + (nu / num_agents) * alpha_agent
    
    optimality_gap_history_gt = []
    current_step_size = step_size
    clip_value = 10.0  # Limite pour le clipping des gradients
    
    for iteration in range(num_iterations):
        new_agent_alphas = [np.zeros(m) for _ in range(num_agents)]
        new_agent_gradients = [np.zeros(m) for _ in range(num_agents)]
        
        # Décroissance du pas si nécessaire (peut être ajustée)
        if iteration > 0 and iteration % 200 == 0:
            current_step_size *= 0.9
        
        for agent_id in range(num_agents):
            # 1. Compute weighted average of alphas
            alpha_avg = np.zeros(m)
            for neighbor_id in range(num_agents):
                if W[agent_id, neighbor_id] > 0:
                    alpha_avg += W[agent_id, neighbor_id] * agent_alphas[neighbor_id]
            
            # 2. Compute current local gradient
            Knm_agent = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            current_grad_local = (sigma**2 / num_agents) * Kmm @ agent_alphas[agent_id] - (Knm_agent.T @ (y_agent - Knm_agent @ agent_alphas[agent_id])) + (nu / num_agents) * agent_alphas[agent_id]
            
            # 3. Apply gradient clipping for stability
            norm_grad = np.linalg.norm(current_grad_local)
            if norm_grad > clip_value:
                current_grad_local = current_grad_local * (clip_value / norm_grad)
            
            # 4. Compute weighted average of gradients
            gradient_avg = np.zeros(m)
            for neighbor_id in range(num_agents):
                if W[agent_id, neighbor_id] > 0:
                    gradient_avg += W[agent_id, neighbor_id] * agent_gradients[neighbor_id]
            
            # 5. Update alpha and gradient tracking variable
            new_agent_alphas[agent_id] = alpha_avg - current_step_size * agent_gradients[agent_id]
            new_agent_gradients[agent_id] = gradient_avg + (current_grad_local - agent_gradients[agent_id])
        
        # 6. Check for divergence
        max_alpha_norm = max([np.linalg.norm(alpha) for alpha in new_agent_alphas])
        if np.isnan(max_alpha_norm) or max_alpha_norm > 1e6:
            print(f"Warning: GT diverged at iteration {iteration}. Reducing step size and continuing...")
            current_step_size *= 0.1
            for agent_id in range(num_agents):
                new_agent_alphas[agent_id] = agent_alphas[agent_id]  # Reset to previous values
                # Recompute with smaller step size
                alpha_avg = np.zeros(m)
                for neighbor_id in range(num_agents):
                    if W[agent_id, neighbor_id] > 0:
                        alpha_avg += W[agent_id, neighbor_id] * agent_alphas[neighbor_id]
                new_agent_alphas[agent_id] = alpha_avg - current_step_size * agent_gradients[agent_id]
        
        agent_alphas = new_agent_alphas
        agent_gradients = new_agent_gradients
        
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_gt.append(optimality_gap)
        
        # 7. Optional early stopping if convergence is detected
        if iteration > 10 and optimality_gap < 1e-6:
            print(f"GT converged at iteration {iteration} with gap: {optimality_gap:.6f}")
            break

    return agent_alphas, optimality_gap_history_gt


def dual_decomposition(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, step_size_dual, num_iterations, alpha_star_centralized):
    """Implémentation standard de la Décomposition Duale (DD) avec une approche plus fondamentale."""
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    
    # Initialize local variables
    agent_alphas = [np.zeros(m) for _ in range(num_agents)]
    agent_lambdas = [np.zeros(m) for _ in range(num_agents)]
    
    # Precompute kernel matrices
    agents_Knm = []
    for agent_x in agents_x_data:
        agents_Knm.append(kernel_matrix(agent_x, X_m_points))

    # For storing history
    optimality_gap_history_dd = []
    
    # For better numerical stability, precompute these matrices
    agents_hessian_inv = []
    for agent_id in range(num_agents):
        Knm = agents_Knm[agent_id]
        # Local Hessian: K_nm^T * K_nm + (regularization terms / num_agents)
        hessian = Knm.T @ Knm + (nu/num_agents) * np.eye(m) + (sigma**2/num_agents) * Kmm
        agents_hessian_inv.append(np.linalg.inv(hessian))
    
    for iteration in range(num_iterations):
        # 1. Primal updates (exact minimization of augmented Lagrangian)
        for agent_id in range(num_agents):
            Knm = agents_Knm[agent_id]
            y_agent = agents_y_data[agent_id]
            
            # Compute right-hand side vector
            rhs = Knm.T @ y_agent + agent_lambdas[agent_id]
            
            # Solve local optimization problem exactly
            agent_alphas[agent_id] = agents_hessian_inv[agent_id] @ rhs
        
        # 2. Compute the global average (standard for dual decomposition)
        z_global = np.mean(agent_alphas, axis=0)
        
        # 3. Dual variable updates
        for agent_id in range(num_agents):
            agent_lambdas[agent_id] = agent_lambdas[agent_id] - step_size_dual * (agent_alphas[agent_id] - z_global)
        
        # Compute optimality gap
        avg_alpha = np.mean(agent_alphas, axis=0)
        optimality_gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_dd.append(optimality_gap)
    
    return agent_alphas, optimality_gap_history_dd, z_global


def admm(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, rho, num_iterations, alpha_star_centralized):
    """
    Réimplémentation "from scratch" d'ADMM en mode pair-à-pair (théorie).
    """
    import itertools
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    
    # Initialisation de alpha, y, lambda
    alpha = [np.zeros(m) for _ in range(num_agents)]
    y = {}
    lambdas = {}
    for (i, j) in itertools.combinations(range(num_agents), 2):
        y[(i, j)] = np.zeros(m)
        lambdas[(i, j)] = np.zeros(m)
    
    # Précompute matrices pour gradient local
    agents_Knm = [kernel_matrix(agents_x_data[i], X_m_points) for i in range(num_agents)]
    
    def local_objective_grad(i, alpha_i):
        # Gradient du coût local (régression kernel)
        Knm_i = agents_Knm[i]
        y_i = agents_y_data[i]
        grad_local = (sigma**2 / num_agents) * Kmm @ alpha_i \
                     - (Knm_i.T @ (y_i - Knm_i @ alpha_i)) \
                     + (nu / num_agents) * alpha_i
        return grad_local
    
    optimality_gap_history_admm = []
    
    for _ in range(num_iterations):
        # 1) Mise à jour locale alpha_i
        for i in range(num_agents):
            # On rassemble la pénalité (alpha_i - y_ij + lambdas/ρ)
            penalty_sum = np.zeros(m)
            neighbors_count = 0
            for j in range(num_agents):
                if j == i: 
                    continue
                # Gestion y_ij et lambda_ij pour indexer (i,j) ou (j,i) de manière unique
                if i < j:
                    penalty_sum += (alpha[i] - y[(i, j)] + lambdas[(i, j)] / rho)
                else:
                    penalty_sum += (alpha[i] - y[(j, i)] - lambdas[(j, i)] / rho)
                neighbors_count += 1
            
            # Simple descente (ou mise à jour) en considérant penalty_sum
            grad_i = local_objective_grad(i, alpha[i]) + rho * penalty_sum
            alpha[i] -= 0.01 * grad_i  # Step-size fixe, ajuster si besoin

        # 2) Mise à jour y_ij
        for (i, j) in y.keys():
            y[(i, j)] = 0.5 * (alpha[i] + alpha[j])

        # 3) Mise à jour lambdas
        for (i, j) in lambdas.keys():
            lambdas[(i, j)] += rho * (alpha[i] - y[(i, j)])
            # Note: on n'a pas besoin de (j, i) car lambda_ij != lambda_ji dans la théorie,
            # mais on stocke seulement (i, j) pour i<j

        # 4) Mesure de l'écart à la solution centralisée
        avg_alpha = np.mean(alpha, axis=0)
        gap = np.linalg.norm(avg_alpha - alpha_star_centralized)
        optimality_gap_history_admm.append(gap)
    
    return alpha, optimality_gap_history_admm, np.mean(alpha, axis=0)


def federated_averaging(agents_X, agents_Y, X_m_points, Kmm, num_rounds, epochs_per_round, batch_size, learning_rate, client_selection_prob=1.0, use_decreasing_lr=False):
    """
    Implémentation améliorée de Federated Averaging (FedAvg).
    
    :param agents_X: Données x par agent
    :param agents_Y: Données y par agent
    :param X_m_points: Points Nyström
    :param Kmm: Matrice kernel entre points Nyström
    :param num_rounds: Nombre de rounds de communication serveur-clients
    :param epochs_per_round: Nombre d'époques locales par round
    :param batch_size: Taille des mini-batchs pour SGD local
    :param learning_rate: Taux d'apprentissage initial
    :param client_selection_prob: Probabilité de sélection d'un client à chaque round (entre 0 et 1)
    :param use_decreasing_lr: Si True, utilise un taux d'apprentissage décroissant O(1/t)
    :return: global_alpha et historique d'erreur
    """
    num_agents = len(agents_X)
    m = len(X_m_points)
    global_alpha = np.zeros(m)

    # Historique pour tracer l'évolution
    objective_error_history_fedavg = []
    
    # Nombre d'échantillons par agent (pour la pondération)
    client_sample_counts = np.array([len(agent_x) for agent_x in agents_X])
    
    # Précalcul des matrices kernel pour éviter de les recalculer à chaque round
    agents_Knm_full = [kernel_matrix(agents_X[i], X_m_points) for i in range(num_agents)]

    for round_num in range(num_rounds):
        # 1. SERVEUR: Sélection des clients avec une probabilité définie
        selected_clients = []
        for agent_id in range(num_agents):
            if np.random.random() < client_selection_prob:
                selected_clients.append(agent_id)
        
        # Si aucun client n'est sélectionné, en sélectionner un aléatoirement
        if not selected_clients:
            selected_clients = [np.random.randint(0, num_agents)]
            
        print(f"Round {round_num+1}/{num_rounds}, {len(selected_clients)}/{num_agents} clients sélectionnés")
        
        # 2. CLIENTS: Mise à jour locale
        local_alphas = []
        sample_counts_selected = []  # Pour la pondération
        
        for agent_id in selected_clients:
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent_full = agents_Knm_full[agent_id]
            local_alpha = global_alpha.copy()
            
            # Apprentissage local sur plusieurs époques
            for epoch in range(epochs_per_round):
                # Déterminer combien de batchs par époque (parcourir potentiellement toutes les données)
                num_samples = len(agent_x)
                num_batches = max(1, num_samples // batch_size)
                
                # Mélanger les indices
                indices_all = np.random.permutation(num_samples)
                
                # Pour chaque batch dans cette époque
                for batch_idx in range(num_batches):
                    # Sélectionner les indices pour ce batch
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    if start_idx >= end_idx:
                        continue
                        
                    indices_batch = indices_all[start_idx:end_idx]
                    
                    # Extraire le batch
                    x_batch = [agent_x[i] for i in indices_batch]
                    y_batch = [agent_y[i] for i in indices_batch]
                    
                    # Calculer la matrice kernel pour ce batch
                    Knm_batch = kernel_matrix(x_batch, X_m_points)
                    
                    # Calculer le gradient local sur ce batch
                    grad_local = (sigma**2 / num_agents) * Kmm @ local_alpha - \
                                (Knm_batch.T @ (y_batch - Knm_batch @ local_alpha)) + \
                                (nu / num_agents) * local_alpha
                    
                    # Taux d'apprentissage décroissant si demandé
                    current_lr = learning_rate
                    if use_decreasing_lr:
                        # Formule O(1/t) où t est le numéro de l'itération global
                        global_iter = round_num * epochs_per_round * num_batches + epoch * num_batches + batch_idx
                        current_lr = learning_rate / (1 + 0.01 * global_iter)
                    
                    # Mise à jour SGD locale
                    local_alpha = local_alpha - current_lr * grad_local
            
            # Stocker le résultat de ce client et son nombre d'échantillons
            local_alphas.append(local_alpha)
            sample_counts_selected.append(client_sample_counts[agent_id])
        
        # 3. SERVEUR: Agrégation pondérée par le nombre d'échantillons
        sample_counts_selected = np.array(sample_counts_selected)
        weights = sample_counts_selected / np.sum(sample_counts_selected)
        
        # Mise à jour pondérée du modèle global
        global_alpha = np.zeros(m)
        for i, alpha in enumerate(local_alphas):
            global_alpha += weights[i] * alpha

        # 4. Calcul de l'erreur objective globale
        objective_error = 0
        for agent_id in range(num_agents):
            agent_x = agents_X[agent_id]
            agent_y = agents_Y[agent_id]
            Knm_agent = agents_Knm_full[agent_id]
            
            # Terme de régularisation L2 (sigma et nu)
            reg_term = (sigma**2 / (2*num_agents)) * global_alpha.T @ Kmm @ global_alpha + \
                       (nu / (2*num_agents)) * np.linalg.norm(global_alpha)**2
            
            # Terme d'erreur quadratique
            error_term = (1/(2*num_agents)) * np.sum((agent_y - Knm_agent @ global_alpha)**2)
            
            objective_error += reg_term + error_term
            
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

    print("--- 0.2. Sélection des points Nyström ---")
    all_indices = list(range(n_total))
    np.random.shuffle(all_indices)
    agents_data_indices = [all_indices[i*points_per_agent:(i+1)*points_per_agent] for i in range(num_agents)]
    agents_x_data = [[x_data[i] for i in indices] for indices in agents_data_indices]
    agents_y_data = [[y_data[i] for i in indices] for indices in agents_data_indices]

    print("--- 0.3 Calcul de la solution centralisée ---")
    alpha_star_centralized, M_indices = solve(x_data[:n_total], y_data[:n_total], selection=True)
    X_m_points = [x_data[i] for i in M_indices]
    y_nystrom = [y_data[i] for i in M_indices]
    Kmm = kernel_matrix(X_m_points, X_m_points)
    print("Solution centralisée calculée.")
    x_prime_grid = np.linspace(-1, 1, nt)
    #visualize_function(x_prime_grid, alpha_star_centralized, X_m_points, "Centralized",  y_nystrom,(x_data[:n_total], y_data[:n_total]))

    print("--- 1. Méthodes Distribuées Classiques (DGD, GT, DD, ADMM) ---")
    print("--- 1.1 Decentralized Gradient Descent (DGD) ---")
    #communication_graph_dgd = nx.Graph()
    #communication_graph_dgd.add_nodes_from(range(num_agents))
    #communication_graph_dgd.add_edges_from([(i, (i+1)%num_agents) for i in range(num_agents)]) # Graphe en anneau
    communication_graph_dgd = nx.complete_graph(num_agents)
    
    step_size_dgd = 0.01
    num_iterations_dgd = 1000

    agent_alphas_dgd, optimality_gap_history_dgd = decentralized_gradient_descent(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dgd,
        step_size_dgd, num_iterations_dgd, alpha_star_centralized
    )

    iterations_dgd = range(num_iterations_dgd)
    #plot_convergence(iterations_dgd, optimality_gap_history_dgd, "DGD")
    avg_alpha_dgd = np.mean(agent_alphas_dgd, axis=0)
    #visualize_function(x_prime_grid, avg_alpha_dgd, X_m_points, "DGD", y_nystrom, (x_data[:n_total], y_data[:n_total]))

    print("--- 1.2 Gradient Tracking (GT) ---")
    communication_graph_gt = communication_graph_dgd.copy()
    step_size_gt = 0.01
    agent_alphas_gt, optimality_gap_history_gt = gradient_tracking(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_gt,
        step_size_gt, num_iterations_dgd, alpha_star_centralized # Réutiliser num_iterations_dgd
    )

    iterations_gt = range(num_iterations_dgd)
    #plot_convergence(iterations_gt, optimality_gap_history_gt, "GradientTracking")
    avg_alpha_gt = np.mean(agent_alphas_gt, axis=0)
    #visualize_function(x_prime_grid, avg_alpha_gt, X_m_points, "GradientTracking",  y_nystrom, (x_data[:n_total], y_data[:n_total]))

    print("--- 1.3 Dual Decomposition (DD) ---")
    communication_graph_dd = communication_graph_dgd.copy()
    step_size_dual_dd = 0.1
    agent_alphas_dd, optimality_gap_history_dd, z_global_dd = dual_decomposition(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dd,
        step_size_dual_dd, num_iterations_dgd, alpha_star_centralized
    )
    
    iterations_dd = range(num_iterations_dgd)
    #plot_convergence(iterations_dd, optimality_gap_history_dd, "DualDecomposition")
    #visualize_function(x_prime_grid, z_global_dd, X_m_points, "DualDecomposition", y_nystrom, (x_data[:n_total], y_data[:n_total])) # Utiliser z_global pour visualiser

    print("--- 1.4 ADMM ---")
    communication_graph_admm = communication_graph_dgd.copy()
    rho_admm = 1.0
    num_iterations_admm = num_iterations_dgd
    agent_alphas_admm, optimality_gap_history_admm, z_global_admm = admm(
        agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, rho_admm,
        num_iterations_admm, alpha_star_centralized
    )

    print(f"Lengths - DGD: {len(optimality_gap_history_dgd)}, GT: {len(optimality_gap_history_gt)}, DD: {len(optimality_gap_history_dd)}, ADMM: {len(optimality_gap_history_admm)}")

    iterations_admm = range(len(optimality_gap_history_admm))
    #plot_convergence(iterations_admm, optimality_gap_history_admm, "ADMM")
    #visualize_function(x_prime_grid, z_global_admm, X_m_points, "ADMM", y_nystrom, (x_data[:n_total], y_data[:n_total]))
    
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
        "DualDecomposition": z_global_dd,
        "ADMM": z_global_admm
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
    visualize_function(x_prime_grid, avg_alpha_dgd_dp, X_m_points, "DGD_DP",  y_nystrom, (x_data[:n_total], y_data[:n_total]))


    print("--- Fin du script principal ---")