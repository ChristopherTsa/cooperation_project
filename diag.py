import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx

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

def decentralized_gradient_descent(agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph, learning_rate, max_iter, alpha_star_centralized):
    num_agents = len(agents_data_indices)
    m = len(X_m_points)
    alpha = {a: np.zeros(m) for a in range(num_agents)}
    optimality_gap_history = []

    for t in range(max_iter):
        gradients = {a: np.zeros(m) for a in range(num_agents)}

        for a in range(num_agents):
            K_nm = kernel_matrix(agents_x_data[a], X_m_points)
            gradients[a] = (sigma**2 / num_agents) * np.dot(Kmm, alpha[a]) + np.dot(K_nm.T, np.dot(K_nm, alpha[a]) - agents_y_data[a]) + (nu / 10) * alpha[a]

        for a in range(num_agents):
            alpha[a] -= learning_rate * gradients[a]
            for neighbor in communication_graph.neighbors(a):
                alpha[a] += learning_rate * (alpha[neighbor] - alpha[a])

        # Enregistrer l'écart d'optimalité à chaque itération
        optimality_gap = np.linalg.norm(alpha[0] - alpha_star_centralized)
        optimality_gap_history.append(optimality_gap)

        if t % 100 == 0:
            print(f"Iteration {t}: Optimality Gap = {optimality_gap}")

    return alpha, optimality_gap_history

def plot_convergence(iterations, optimality_gap_history, title):
    plt.figure()
    plt.plot(iterations, optimality_gap_history, label=title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality Gap')
    plt.title(f'Convergence of {title}')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_function(x_prime_grid, avg_alpha, X_m_points, title):
    f_values = np.zeros(len(x_prime_grid))
    for i, x_prime in enumerate(x_prime_grid):
        f_values[i] = np.sum(avg_alpha * kernel_matrix(np.array([x_prime]), X_m_points))

    plt.figure()
    plt.plot(x_prime_grid, f_values, label=title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Function Visualization for {title}')
    plt.legend()
    plt.grid()
    plt.show()

# --- Fonction pour résoudre le problème centralisé ---
def centralized_solution(Kmm, Knm, y, sigma, nu):
    """Calcule la solution centralisée pour comparer les méthodes distribuées."""
    m = Kmm.shape[0]
    alpha_star = np.linalg.solve(sigma**2 * Kmm + Knm.T @ Knm + nu * np.eye(m), Knm.T @ y)
    return alpha_star

# --- 1. Decentralized Gradient Descent (DGD) ---
print("--- Decentralized Gradient Descent (DGD) ---")

# --- Paramètres Globaux ---
sigma = 0.5
nu = 1.0

# 1. Charger les données
data_part1 = load_data('data/first_database.pkl')
x_data, y_data = data_part1[0], data_part1[1]

n_total = 100
m_nystrom = 10
num_agents = 5
points_per_agent = n_total // num_agents

X_m_points, M_indices = nystrom_approximation(x_data[:n_total], m_nystrom)

agents_data_indices = [list(range(i*points_per_agent, (i+1)*points_per_agent)) for i in range(num_agents)]
agents_x_data = [[x_data[i] for i in indices] for indices in agents_data_indices]
agents_y_data = [[y_data[i] for i in indices] for indices in agents_data_indices]

Kmm = kernel_matrix(X_m_points, X_m_points)

print("Calcul de la solution centralisée...")
Knm_centralized = kernel_matrix(x_data[:n_total], X_m_points)
alpha_star_centralized = centralized_solution(Kmm, Knm_centralized, y_data[:n_total], sigma, nu)
print("Solution centralisée calculée.")

# Graphe de communication en anneau
communication_graph_dgd = nx.Graph()
communication_graph_dgd.add_nodes_from(range(num_agents))
communication_graph_dgd.add_edges_from([(i, (i+1)%num_agents) for i in range(num_agents)])

# Paramètres DGD
step_size_dgd = 0.01
num_iterations_dgd = 1000

# Exécution de DGD
agent_alphas_dgd, optimality_gap_history_dgd = decentralized_gradient_descent(
    agents_data_indices, agents_x_data, agents_y_data, X_m_points, Kmm, communication_graph_dgd,
    step_size_dgd, num_iterations_dgd, alpha_star_centralized
)

# Tracer la convergence
iterations_dgd = np.linspace(0, num_iterations_dgd, len(optimality_gap_history_dgd))    
plot_convergence(iterations_dgd, optimality_gap_history_dgd, "DGD")

#print(agent_alphas_dgd)
#print(len(agent_alphas_dgd))    

# Visualiser la fonction
# Convertir les valeurs du dictionnaire en une liste de tableaux
alpha_list = list(agent_alphas_dgd.values())

# Calculer la moyenne des alphas sur tous les agents
avg_alpha_dgd = np.mean(alpha_list, axis=0)

# Tracer la convergence
iterations_dgd = range(num_iterations_dgd)
plot_convergence(iterations_dgd, optimality_gap_history_dgd, "DGD")

# Visualiser la fonction
x_prime_grid = np.linspace(-1, 1, 250)
visualize_function(x_prime_grid, avg_alpha_dgd, X_m_points, "DGD")

