"""linalg_tutorial 套件的單元測試：所有自製演算法都與 numpy/scipy 對照。"""

from __future__ import annotations

import numpy as np
import pytest

from linalg_tutorial import determinant, eigen, elimination, orthogonal, subspaces, svd_tools
from linalg_tutorial.utils import allclose

RNG = np.random.default_rng(12345)


def random_matrix(m, n, seed=0):
    return np.random.default_rng(seed).standard_normal((m, n))


# --------------------------------------------------------------------------
# elimination
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_plu_factorization(n):
    A = random_matrix(n, n, seed=n)
    P, L, U = elimination.plu(A)
    assert allclose(P @ A, L @ U, 1e-10)
    assert allclose(np.tril(L), L, 1e-12)            # L 下三角
    assert allclose(np.triu(U), U, 1e-12)            # U 上三角
    assert np.all(np.abs(L) <= 1 + 1e-12)            # 部分軸選取的保證


@pytest.mark.parametrize("n", [2, 6, 15])
def test_solve_matches_numpy(n):
    A = random_matrix(n, n, seed=n + 100)
    b = random_matrix(n, 1, seed=n + 200).ravel()
    assert allclose(elimination.solve(A, b), np.linalg.solve(A, b), 1e-8)


def test_lu_no_pivot_raises_on_zero_pivot():
    with pytest.raises(ValueError):
        elimination.lu_no_pivot(np.array([[0.0, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize("n", [1, 3, 8])
def test_cholesky(n):
    M = random_matrix(n, n, seed=n + 7)
    S = M @ M.T + n * np.eye(n)
    R = elimination.cholesky(S)
    assert allclose(R.T @ R, S, 1e-9)
    assert allclose(np.triu(R), R, 1e-12)


def test_cholesky_rejects_indefinite():
    with pytest.raises(ValueError):
        elimination.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("n", [2, 5, 10])
def test_inverse_gauss_jordan(n):
    A = random_matrix(n, n, seed=n + 31) + n * np.eye(n)
    assert allclose(elimination.inverse_gauss_jordan(A), np.linalg.inv(A), 1e-8)


def test_second_difference_matrix_properties():
    K = elimination.second_difference_matrix(6)
    assert allclose(K, K.T, 1e-15)
    assert np.all(np.linalg.eigvalsh(K) > 0)         # 正定
    assert np.isclose(np.linalg.det(K), 7.0)         # det Kₙ = n + 1


# --------------------------------------------------------------------------
# subspaces
# --------------------------------------------------------------------------
def test_rref_and_cr_factorization():
    A = np.array([[1.0, 3.0, 4.0], [2.0, 4.0, 2.0], [3.0, 7.0, 6.0]])
    R0, piv = subspaces.rref(A)
    assert piv == [0, 1]
    C, R = subspaces.cr_factor(A)
    assert allclose(C @ R, A, 1e-12)
    assert C.shape == (3, 2) and R.shape == (2, 3)


@pytest.mark.parametrize("seed", range(5))
def test_rank_matches_numpy_for_well_conditioned(seed):
    rng = np.random.default_rng(seed)
    r = rng.integers(1, 4)
    A = rng.standard_normal((6, r)) @ rng.standard_normal((r, 5))
    assert subspaces.rank_svd(A) == np.linalg.matrix_rank(A)


def test_nullspace_is_a_basis():
    A = np.array([[1.0, 7.0, 3.0, 3.0], [2.0, 14.0, 6.0, 70.0],
                  [2.0, 14.0, 9.0, 97.0]])
    N = subspaces.nullspace(A)
    assert N.shape == (4, 4 - subspaces.rank(A))
    assert allclose(A @ N, np.zeros((3, N.shape[1])), 1e-10)
    assert subspaces.is_independent(N)


def test_four_subspace_dimensions():
    A = random_matrix(5, 7, seed=3) @ random_matrix(7, 7, seed=4)
    rep = subspaces.four_subspace_report(A)
    m, n = A.shape
    r = rep["rank"]
    assert rep["dims"] == {"C(A)": r, "N(A)": n - r, "C(A^T)": r, "N(A^T)": m - r}
    assert rep["column_space"].shape == (m, r)
    assert rep["nullspace"].shape == (n, n - r)


def test_complete_solution_consistent_and_inconsistent():
    A = np.array([[1.0, 3.0, 0.0, 2.0], [0.0, 0.0, 1.0, 4.0],
                  [1.0, 3.0, 1.0, 6.0]])
    ok = subspaces.complete_solution(A, np.array([1.0, 6.0, 7.0]))
    bad = subspaces.complete_solution(A, np.array([1.0, 6.0, 8.0]))
    assert ok["consistent"] and allclose(A @ ok["particular"],
                                         np.array([1.0, 6.0, 7.0]), 1e-10)
    assert not bad["consistent"]


# --------------------------------------------------------------------------
# determinant
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_determinant_methods_agree(n):
    A = random_matrix(n, n, seed=n + 55)
    d_np = np.linalg.det(A)
    assert np.isclose(determinant.det_lu(A), d_np, rtol=1e-8, atol=1e-10)
    assert np.isclose(determinant.det_cofactor(A), d_np, rtol=1e-8, atol=1e-10)
    terms = determinant.big_formula_terms(A)
    assert len(terms) == np.math.factorial(n) if hasattr(np, "math") else True
    assert np.isclose(sum(s * p for _, s, p in terms), d_np, rtol=1e-8, atol=1e-10)


def test_cofactor_inverse_and_cramer():
    A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, -1.0])
    assert allclose(determinant.inverse_by_cofactors(A), np.linalg.inv(A), 1e-10)
    assert allclose(determinant.cramer(A, b), np.linalg.solve(A, b), 1e-10)


def test_volume_equals_abs_det():
    E = random_matrix(4, 4, seed=9)
    assert np.isclose(determinant.volume(E), abs(np.linalg.det(E)), rtol=1e-9)


# --------------------------------------------------------------------------
# orthogonal
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(5, 3), (8, 8), (10, 4)])
def test_householder_qr(shape):
    A = random_matrix(*shape, seed=sum(shape))
    Q, R = orthogonal.householder_qr(A)
    assert allclose(Q @ R, A, 1e-9)
    assert allclose(Q.T @ Q, np.eye(Q.shape[1]), 1e-10)
    assert allclose(np.triu(R), R, 1e-12)


@pytest.mark.parametrize("shape", [(6, 3), (9, 5)])
def test_gram_schmidt_orthonormal(shape):
    A = random_matrix(*shape, seed=sum(shape) + 1)
    Q = orthogonal.gram_schmidt(A)
    assert orthogonal.is_orthonormal(Q)


def test_projection_matrix_properties():
    A = random_matrix(7, 3, seed=21)
    P = orthogonal.projection_matrix(A)
    assert allclose(P @ P, P, 1e-10)
    assert allclose(P.T, P, 1e-10)
    assert np.isclose(np.trace(P), 3.0)


@pytest.mark.parametrize("shape", [(10, 3), (3, 10), (6, 6)])
def test_pseudoinverse_penrose_conditions(shape):
    A = random_matrix(*shape, seed=sum(shape) + 2)
    Ap = orthogonal.pseudoinverse(A)
    assert allclose(A @ Ap @ A, A, 1e-9)
    assert allclose(Ap @ A @ Ap, Ap, 1e-9)
    assert allclose((A @ Ap).T, A @ Ap, 1e-9)
    assert allclose((Ap @ A).T, Ap @ A, 1e-9)


def test_least_squares_matches_numpy():
    A = random_matrix(12, 4, seed=77)
    b = random_matrix(12, 1, seed=78).ravel()
    expected = np.linalg.lstsq(A, b, rcond=None)[0]
    assert allclose(orthogonal.least_squares_normal(A, b), expected, 1e-8)
    assert allclose(orthogonal.least_squares_qr(A, b), expected, 1e-8)


# --------------------------------------------------------------------------
# eigen
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 6, 10])
def test_qr_algorithm_and_jacobi(n):
    M = random_matrix(n, n, seed=n + 300)
    S = M + M.T
    expected = np.sort(np.linalg.eigvalsh(S))
    assert allclose(np.sort(eigen.qr_algorithm(S)[0]), expected, 1e-6)
    lam, V = eigen.jacobi_eigen(S)
    assert allclose(np.sort(lam), expected, 1e-9)
    assert allclose(V @ np.diag(lam) @ V.T, S, 1e-9)


def test_power_and_inverse_power_iteration():
    S = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    lam_all = np.linalg.eigvalsh(S)
    lam_max, v, _ = eigen.power_iteration(S)
    assert np.isclose(lam_max, lam_all[-1], rtol=1e-8)
    assert allclose(S @ v, lam_max * v, 1e-6)
    lam_near, _ = eigen.inverse_power_iteration(S, shift=2.1)
    assert np.isclose(lam_near, lam_all[np.argmin(np.abs(lam_all - 2.1))], rtol=1e-8)


@pytest.mark.parametrize("t", [0.0, 0.5, 2.0])
def test_matrix_exponential_real_and_complex(t):
    A = random_matrix(4, 4, seed=5) * 0.3
    assert allclose(eigen.matrix_exp_series(A, t, 90),
                    eigen.matrix_exp_by_eigen(A, t), 1e-9)
    H = np.array([[1.0, 1j], [-1j, 2.0]])            # Hermitian ⇒ e^{-iHt} unitary
    U = eigen.matrix_exp_series(-1j * H, t, 90)
    assert allclose(U.conj().T @ U, np.eye(2, dtype=complex), 1e-9)


def test_matrix_exp_of_jordan_block():
    J = np.array([[2.0, 1.0], [0.0, 2.0]])
    E = eigen.matrix_exp_series(J, 1.0, 80)
    assert allclose(E, np.array([[np.e ** 2, np.e ** 2], [0.0, np.e ** 2]]), 1e-10)


def test_generalized_symmetric_eig():
    A = np.array([[6.0, 3.0], [3.0, 2.0]])
    B = np.array([[6.0, 0.0], [0.0, 3.0]])
    lam, X = eigen.generalized_symmetric_eig(B, A)
    assert allclose(np.sort(lam), np.sort([5 - np.sqrt(19), 5 + np.sqrt(19)]), 1e-9)
    assert allclose(X.T @ A @ X, np.eye(2), 1e-9)     # A-正交歸一
    for k in range(2):
        assert allclose(B @ X[:, k], lam[k] * (A @ X[:, k]), 1e-9)


def test_positive_definite_tests_agree():
    S = np.array([[2.0, 4.0], [4.0, 9.0]])
    assert eigen.is_positive_definite(S)
    assert not eigen.is_positive_definite(np.array([[9.0, 3.0], [3.0, 1.0]]))
    assert not eigen.is_positive_definite(np.array([[0.0, 2.0], [2.0, 1.0]]))


def test_markov_steady_state():
    P = np.array([[0.8, 0.3], [0.2, 0.7]])
    pi = eigen.markov_steady_state(P)
    assert allclose(P @ pi, pi, 1e-10)
    assert np.isclose(pi.sum(), 1.0)
    assert allclose(pi, np.array([0.6, 0.4]), 1e-8)


# --------------------------------------------------------------------------
# svd_tools
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_svd_from_eigen(shape):
    A = random_matrix(*shape, seed=sum(shape) + 11)
    U, s, Vt = svd_tools.svd_from_eigen(A)
    assert allclose(np.sort(s), np.sort(np.linalg.svd(A, compute_uv=False)), 1e-9)
    assert allclose(U @ np.diag(s) @ Vt, A, 1e-9)


def test_low_rank_approx_is_eckart_young():
    A = random_matrix(8, 6, seed=42)
    s = np.linalg.svd(A, compute_uv=False)
    for k in range(1, 6):
        Ak = svd_tools.low_rank_approx(A, k)
        assert np.linalg.matrix_rank(Ak, tol=1e-10) == k
        assert np.isclose(svd_tools.frobenius_error(A, Ak),
                          np.sqrt(np.sum(s[k:] ** 2)), rtol=1e-9)


def test_pca_matches_covariance_eigen():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 4))
    res = svd_tools.pca(X, n_components=4)
    lam = np.linalg.eigvalsh(res["covariance"])[::-1]
    assert allclose(res["explained_variance"], lam, 1e-8)
    assert np.isclose(res["explained_ratio"].sum(), 1.0)


def test_polar_and_condition_number():
    A = np.array([[5.0, 4.0], [0.0, 3.0]])
    Q, S = svd_tools.polar_decomposition(A)
    assert allclose(Q @ S, A, 1e-10)
    assert allclose(Q.T @ Q, np.eye(2), 1e-10)
    assert np.all(np.linalg.eigvalsh(S) > 0)
    s = np.linalg.svd(A, compute_uv=False)
    assert np.isclose(svd_tools.condition_number(A), s[0] / s[-1])
