import numpy as np

from TsaKwetShinMaiGeffray import kernel_matrix, plot_convergence, visualize_function


sigma = 0.5 # Variance du bruit
nu = 1  # Paramètre de régularisation

def diag_incremental(x_data, y_data, X_m_points, Kmm, num_iterations, step_size, alpha_star_centralized=None, verbose=False):
    """
    Implémentation de l'algorithme DIAG (Double Incremental Aggregated Gradient) pour la régression par noyaux.
    
    Paramètres:
      - x_data : liste des points d'entrée (utilisés pour construire K_nm)
      - y_data : liste des étiquettes correspondantes
      - X_m_points : points sélectionnés pour l’approximation de Nyström
      - Kmm : matrice de noyaux évaluée sur X_m_points
      - num_iterations : nombre total d’itérations de l’algorithme
      - step_size : pas utilisé dans la mise à jour (par exemple, 2/(μ+L))
      - alpha_star_centralized : solution centralisée (pour suivi de la convergence), optionnel
      - verbose : affiche des informations toutes les 100 itérations
      
    Renvoie:
      - alpha : vecteur solution final (dimension m)
      - optimality_gap_history : historique de l’écart optimal (||alpha - alpha_star_centralized||) par itération
    """
    n = len(x_data)  # Nombre total de données
    m = len(X_m_points)
    
    # Calcul de la matrice K_nm (dimensions n x m)
    Knm = kernel_matrix(x_data, X_m_points)
    
    # TEST 1 : Vérifier les statistiques de la matrice de noyaux
    print("=== TEST 1 : Statistiques de K_nm ===")
    print("K_nm min: {:.6f}, max: {:.6f}, moyenne: {:.6f}".format(Knm.min(), Knm.max(), Knm.mean()))
    
    # Initialisation de alpha (solution)
    alpha = np.zeros(m)
    
    # Initialisation de la mémoire pour chaque fonction f_i
    y_memory = [alpha.copy() for _ in range(n)]
    grad_memory = []
    for i in range(n):
        # Calcul du gradient pour f_i :
        # grad_f_i(alpha) = 1/n * (sigma^2 * Kmm @ alpha + nu * alpha)
        #                   - 1/n * (Knm[i] * (y_data[i] - np.dot(Knm[i], alpha)))
        grad_i = (sigma**2 * (Kmm @ alpha) + nu * alpha) / n - (1/n) * np.array(Knm[i]) * (y_data[i] - np.dot(Knm[i], alpha))
        grad_memory.append(grad_i)
    print("=== TEST 2 : Premier gradient initial ===")
    print("grad_memory[0]:", grad_memory[0])
    
    # Initialisation des agrégats
    v = n * alpha.copy()         # v^0 = n * alpha^0
    g = np.sum(grad_memory, axis=0)  # g^0 = somme de tous les gradients évalués aux y_memory
    
    print("=== TEST 3 : Agrégats initiaux ===")
    print("v initial:", v)
    print("g initial:", g)
    
    optimality_gap_history = []
    
    for k in range(num_iterations):
        # Mise à jour de alpha selon : alpha_{k+1} = (v - step_size * g) / n
        alpha_new = (v - step_size * g) / n
        
        # Sélection cyclique de l'indice i
        i = k % n
        
        # Mise à jour de l'agrégat v en remplaçant la contribution de y_memory[i]
        v = v + (alpha_new - y_memory[i])
        
        # Calcul du nouveau gradient pour f_i évaluée en alpha_new
        r = np.dot(Knm[i], alpha_new)
        grad_new = (sigma**2 * (Kmm @ alpha_new) + nu * alpha_new) / n - (1/n) * np.array(Knm[i]) * (y_data[i] - r)
        
        # Mise à jour de l'agrégat g
        g = g + (grad_new - grad_memory[i])
        
        # Mise à jour de la mémoire pour f_i
        y_memory[i] = alpha_new.copy()
        grad_memory[i] = grad_new.copy()
        
        # Actualisation de alpha
        alpha = alpha_new.copy()
        
        # Suivi de la convergence
        if alpha_star_centralized is not None:
            gap = np.linalg.norm(alpha - alpha_star_centralized)
            optimality_gap_history.append(gap)
        else:
            optimality_gap_history.append(np.linalg.norm(alpha))
        
        if verbose and (k % 100 == 0):
            print(f"DIAG - Iteration {k+1}/{num_iterations}")
            print("  Norme de alpha:", np.linalg.norm(alpha))
            print("  Agrégat v:", v)
            print("  Agrégat g:", g)
            if alpha_star_centralized is not None:
                print("  Optimality gap:", optimality_gap_history[-1])
    
    return alpha, optimality_gap_history


if __name__ == "__main__":
   
    from TsaKwetShinMaiGeffray import load_data, visualize_data, plot_convergence, visualize_function, solve
    
    # Paramètres pour l'exemple
    n_total = 100
    nt = 250
   
    data = load_data('data/first_database.pkl')
    x_data, y_data = data[0], data[1]
    
    # Calcul de l'approximation de Nyström
    import math as m
    m_nystrom = m.ceil(m.sqrt(n_total))
    sel = list(range(n_total))
    ind = np.random.choice(sel, m_nystrom, replace=False)
    X_m_points = [x_data[i] for i in ind]
    y_nystrom = [y_data[i] for i in ind]  # Points pour visualisation
    Kmm = kernel_matrix(X_m_points, X_m_points)
    
    # Calcul de la solution centralisée (pour comparaison)
    alpha_star_centralized, _ = solve(x_data[:n_total], y_data[:n_total], selection=True)
    
    # Calcul de la matrice K_nm pour déterminer μ et L
    Knm = kernel_matrix(x_data[:n_total], X_m_points)
    m_dim = len(X_m_points)
    # La Hessienne H = sigma^2*Kmm + K_nm^T*K_nm + nu*I
    H = sigma**2 * Kmm + np.dot(Knm.T, Knm) + nu * np.eye(m_dim)
    eigvals = np.linalg.eigvals(H)
    mu = np.min(eigvals.real)
    L = np.max(eigvals.real)
    epsilon = 0.1
    print("Calculated mu =", mu)
    print("Calculated L =", L)
    print("Step size (epsilon) set to 2/(mu+L) =", epsilon)
    
    num_iterations_diag = 100000
    step_size_diag = epsilon  
    
    print("--- Exécution de DIAG avec tests de débogage ---")
    alpha_diag, gap_history_diag = diag_incremental(
        x_data[:n_total],
        y_data[:n_total],
        X_m_points,
        Kmm,
        num_iterations_diag,
        step_size_diag,
        alpha_star_centralized=alpha_star_centralized,
        verbose=True
    )
    
    iterations_diag = range(num_iterations_diag)
    
    plot_convergence(iterations_diag, gap_history_diag, "DIAG")
    
    x_prime_grid = np.linspace(-1, 1, nt)
    visualize_function(x_prime_grid, alpha_diag, X_m_points, "DIAG", y_nystrom, (x_data[:n_total], y_data[:n_total]))
    
    print("--- Fin du script DIAG avec tests ---")
