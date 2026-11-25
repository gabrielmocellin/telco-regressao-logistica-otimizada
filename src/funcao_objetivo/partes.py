"""
Implementação da modelagem matemática da Função Objetivo e suas derivadas
(1ª e 2ª ordem) para Regressão Logística:
  - sigmoide(z)
  - funcao_objetivo(w, X, y)
  - gradiente(w, X, y)
  - hessiana(w, X, y)
"""
import numpy as np


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Calcula a função sigmóide sigma(z) = 1 / (1 + exp(-z)) de forma numericamente estável."""
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(
        z_clip >= 0,
        1.0 / (1.0 + np.exp(-z_clip)),
        np.exp(z_clip) / (1.0 + np.exp(z_clip)),
    )


def funcao_objetivo(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula o valor da função objetivo f(w):
    Log-verossimilhança negativa média (entropia cruzada binária / Log-Loss).
    """
    z = X @ w
    # Forma vetorizada e numericamente estável (sem overflow em exp):
    #   z >= 0 -> z - z*y + ln(1 + exp(-z))
    #   z <  0 ->   - z*y + ln(1 + exp(z))
    perda_amostras = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(perda_amostras))


def gradiente(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Calcula o vetor gradiente nabla f(w) in R^n:
      nabla f(w) = (1/m) * X^T @ (sigma(X @ w) - y)
    """
    m = X.shape[0]
    p = sigmoide(X @ w)
    return (X.T @ (p - y)) / m


def hessiana(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Calcula a matriz Hessiana nabla^2 f(w) in R^(n x n):
      nabla^2 f(w) = (1/m) * X^T @ D @ X
    onde D = diag(p_i * (1 - p_i)) e p = sigma(X @ w).
    """
    m = X.shape[0]
    p = sigmoide(X @ w)
    d = p * (1.0 - p)
    return (X.T @ (d[:, np.newaxis] * X)) / m
