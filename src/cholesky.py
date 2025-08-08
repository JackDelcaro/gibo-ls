import torch

def schur_complement_inverse(A11_inv, A12, A21, A22):
    """
    Compute the inverse of a block matrix given the inverse of the top-left block using Cholesky factorization.
    
    Arguments:
    A11_inv : torch.Tensor : Inverse of the top-left block A11
    A12     : torch.Tensor : Top-right block of A
    A21     : torch.Tensor : Bottom-left block of A
    A22     : torch.Tensor : Bottom-right block of A
    
    Returns:
    A_inv   : torch.Tensor : Inverse of the full block matrix A using Cholesky factorization.
    """
    # Step 1: Compute the Schur complement S
    S = A22 - A21 @ A11_inv @ A12

    # Step 2: Compute the Cholesky factor of the Schur complement S
    L_s = torch.linalg.cholesky(S)

    # Step 3: Use Cholesky factor to solve for S_inv
    identity = torch.eye(L_s.size(0), dtype=L_s.dtype, device=L_s.device)
    S_inv = torch.cholesky_solve(identity, L_s)

    # Step 4: Compute the blocks of the inverse matrix
    B11 = A11_inv + A11_inv @ A12 @ S_inv @ A21 @ A11_inv
    B12 = -A11_inv @ A12 @ S_inv
    B21 = -S_inv @ A21 @ A11_inv
    B22 = S_inv

    # Step 5: Construct the inverse matrix A_inv
    A_inv = torch.cat((torch.cat((B11, B12), dim=1), torch.cat((B21, B22), dim=1)), dim=0)
    
    return A_inv

def one_step_cholesky(
    top_left: torch.Tensor, K_Xθ: torch.Tensor, K_θθ: torch.Tensor, A_inv: torch.Tensor
) -> torch.Tensor:
    '''Update the Cholesky factor when the matrix is extended.

    Note: See thesis appendix A.2 for notation of args and further information.

    Args:
        top_left: Cholesky factor L11 of old matrix A11.
        K_Xθ: Upper right bock matrix A12 of new matrix A.
        K_θθ: Lower right block matrix A22 of new matrix A.
        A_inv: Inverse of old matrix A11.

    Returns:
        New cholesky factor S of new matrix A.
    '''
    # Solve with A \ b: A @ x = b, x = A^(-1) @ b,
    # top_right = L11^T \ A12 = L11^T  \ K_Xθ, top_right = (L11^T)^(-1) @ K_Xθ,
    # Use: (L11^(-1))^T = L11 @ A11^(-1).
    # Hint: could also be solved with torch.cholesky_solve (in theory faster).
    top_right = top_left @ (A_inv @ K_Xθ)
    bot_left = torch.zeros_like(top_right).transpose(-1, -2)
    bot_right = torch.linalg.cholesky(
        K_θθ - top_right.transpose(-1, -2) @ top_right, upper=True
    )
    return torch.cat(
        [
            torch.cat([top_left, top_right], dim=-1),
            torch.cat([bot_left, bot_right], dim=-1),
        ],
        dim=-2,
    )
