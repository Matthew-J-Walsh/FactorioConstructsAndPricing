from globalsandimports import *

import sympy

def farkas_lemma(A: np.ndarray, b: np.ndarray, solver: Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray[Real]]) -> None:
    """
    Throws an error if solver violates Farkas lemma on an A and b.
    https://en.wikipedia.org/wiki/Farkas%27_lemma

    Parameters
    ----------
    A:
        A matrix
    b:
        A vector
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x=b, x>=0, minimize c*x. Return x. 
        If it cannot solve the problem for whatever reason it should return None.
    """
    try:
        x = solver(A, b)
        assert not x is None
    except:
        A_dual = np.concatenate([A.T, np.identity(A.shape[0])])
        x = solver(A_dual, np.zeros(A_dual.shape[0]), b.T)
        c = np.dot(x, b.T)
        assert c < 0

def rank_analysis(A: np.ndarray[Fraction]) -> None:
    """
    Determines the handed-ness of a linear programming problem.

    Parameters
    ----------
    A:
        A matrix
    b:
        A vector
    """
    A_rank = np.linalg.matrix_rank(A.astype(np.double))
    AT_rank = np.linalg.matrix_rank(A.astype(np.double))
    if A.shape[0]==A.shape[1] and A_rank==A.shape[0]: #square matrix
        logging.info("A is a square invertable matrix.")
    elif A_rank==A.shape[0]: #linearly independent rows
        logging.info("A has a left-handed inverse.")
    elif AT_rank==A.shape[1]: #linearly independent columns
        logging.info("A has a right-handed inverse.")
    else:
        logging.info("No linear independence found. Please give an A matrix with linear independent rows or columns.")

def inverse_analysis(A: np.ndarray[Fraction], b: np.ndarray) -> bool:
    """
    Uses the Moore-Penrose inverse to analyze the feasiblity of Ax=b.
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    Method spent 6 hours on a 400x400 matrix so basically its unusable.
    Cant use numpy/scipy methods because they are inaccurate.

    Parameters
    ----------
    A:
        A matrix
    b:
        A vector
    
    Returns
    -------
    False iff problem is always infesible or unsolvable.
    """
    raise DeprecationWarning

    is_left, is_right = False, False
    A = sympy.nsimplify(sympy.Matrix(A), rational=True)
    if A.shape[0]==A.shape[1] and len(A.rref()[1])==A.shape[0]: #square matrix
        is_left, is_right = True, True
        A_inv = A.inv('ADJ')
        #assert (A_inv @ A).shape[0]==A.shape[0] and (A_inv @ A).shape[1]==A.shape[1]
        assert (A_inv @ A == sympy.eye(A.shape[0])), A_inv @ A
        assert (A @ A_inv == sympy.eye(A.shape[0])), A @ A_inv
    elif len(A.rref()[1])==A.shape[0]: #linearly independent rows
        is_left = True
        A_inv = A.T @ ((A @ A.T) ** -1)
        #assert (A @ A_inv).shape[0]==A.shape[0] and (A @ A_inv).shape[1]==A.shape[0]
        assert (A @ A_inv == sympy.eye(A.shape[0])), A @ A_inv
    elif len(A.T.rref()[1])==A.shape[1]: #linearly independent columns
        is_right = True
        A_inv = ((A.T @ A) ** -1) @ A.T
        #assert (A_inv @ A).shape[0]==A.shape[1] and (A_inv @ A).shape[1]==A.shape[1]
        assert (A_inv @ A == sympy.eye(A.shape[1])), A_inv @ A
    else:
        logging.warning("No linear independence found. Please give an A matrix with linear independent rows or columns.")
        return True

    if is_left and is_right:
        logging.info("A is a square invertable matrix yielding 1 possible solution.")
        if (A_inv @ b >= 0).all():
            logging.info("\tx>=0")
            return True
        else:
            logging.info("\tx is not in the positive orthant. Problem is infesible.")
            return False
    elif is_right:
        logging.info("A has a right-handed inverse. There is at least 1 possible solution.")
        logging.info("\tIndeterminate feasibility.")
        return True
    else:
        if is_left:
            logging.info("A has a left-handed inverse. Checking if there are solutions.")
        else:
            logging.info("A has no left/right-handed inverses. Checking if there are solutions.")
        if (A @ A_inv @ b == b).all():
            logging.info("\tSystem is consistent, there are solutions of indeterminate feasiblity")
            return True
        else:
            logging.info("\tSystem is inconsistent, there are no possible solutions.")
            return False
        