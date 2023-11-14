import numpy as np
from scipy.linalg import cholesky, qr, svd

def randomized_nystrom(A, sketching_matrix, k):
    # Step 1: Generate a random sketching matrix Ω_1
    n = A.shape[0]
    
    # Step 2: Compute C = AΩ_1
    C = A@sketching_matrix
    
    # Step 3: Compute B = Ω_1^T C and its Cholesky factorization B = LL^T
    B = sketching_matrix.T@C
    L = cholesky(B, lower=True)
    
    # Step 4: Compute Z = CL^(-T) (where L^(-T) is the transpose of the inverse of L)
    Z = C@np.linalg.inv(L).T
    
    # Step 5: Compute the QR factorization of Z = QR
    Q, R = qr(Z, mode='economic')
    
    # Step 6: Compute the SVD factorization of R = U~ΣV~^T
    U_tilde, Sigma, V_tilde_transposed = svd(R, full_matrices=False)
    # Truncate the SVD
    U_tilde = U_tilde[:,:k]
    Sigma = Sigma[:k]
    V_tilde = V_tilde_transposed[:k,:].T
    
    # Step 7: Compute Û = QÛ~
    U_hat = Q@U_tilde
    
    # Step 8: Output the factorization ÛΣ^2Û^T
    A_nystrom = U_hat.dot(np.diag(Sigma**2)).dot(U_hat.T)
    
    return A_nystrom

if __name__ == "__main__":
    # Example usage:
    # Define a random matrix A
    n = 20  # size of the matrix
    l = 10
    k = 5   # rank of the approximation
    A = np.random.randn(n, n)
    A = A + np.identity(n)*2
    A = A @ A.T
    sketching_matrix = np.random.randn(n, l)
    
    # Call the function
    A_nystrom_approx = randomized_nystrom(A, sketching_matrix, k)
    print(np.linalg.norm(A_nystrom_approx-A))

    sketching_matrix = np.random.randn(n, n)
    A_nystrom_approx = randomized_nystrom(A, sketching_matrix, k)
    print(np.linalg.norm(A_nystrom_approx-A))
