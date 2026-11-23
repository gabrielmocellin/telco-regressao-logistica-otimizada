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
    # Limita valores de z para evitar overflow/underflow em exp(-z) entre -500 e 500
    # é um intervalo seguro para a função exponencial.
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(
        z_clip >= 0,
        1.0 / (1.0 + np.exp(-z_clip)),
        np.exp(z_clip) / (1.0 + np.exp(z_clip)),
    )


# Explicação breve da função objetivo:
# - w: pesos
# - X: matriz de features
# - y: vetor de labels
def funcao_objetivo(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calcula o valor da função objetivo f(w):
    Log-verossimilhança negativa média (entropia cruzada binária / Log-Loss).
    """
    # Resultado (z): É um vetor onde cada posição contém o valor da pontuação linear (logit)
    z = X @ w  # multiplicação matricial (ou produto escalar).

    # Cálculo da Log-Loss (Entropia Cruzada Binária): -[y*ln(p) + (1-y)*ln(1-p)]
    # Substituindo p = 1 / (1 + exp(-z)) e simplificando os logaritmos, evitamos
    # calcular ln(0). Para que np.exp() nunca receba um expoente positivo gigante
    # (o que causaria overflow), separamos pelo sinal de z[i]:
    perda_amostras = np.zeros_like(z, dtype=float)

    for i in range(len(z)):
        if z[i] >= 0:
            # Caso 1 (z >= 0): usamos exp(-z[i]) para que o expoente seja negativo (seguro)
            # Fórmula simplificada: z - z*y + ln(1 + exp(-z))
            perda_amostras[i] = z[i] - z[i] * y[i] + np.log1p(np.exp(-z[i]))
        else:
            # Caso 2 (z < 0): usamos exp(z[i]) pois z[i] já é negativo (seguro)
            # Fórmula simplificada: -z*y + ln(1 + exp(z))
            perda_amostras[i] = -z[i] * y[i] + np.log1p(np.exp(z[i]))

    perda_media = float(np.mean(perda_amostras))
    return perda_media


def gradiente(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Calcula o vetor gradiente nabla f(w) in R^n:
      nabla f(w) = (1/m) * X^T @ (sigma(X @ w) - y)
    """
    m = X.shape[0] # Número de clientes (7043)
    p = sigmoide(X @ w)  # Vetor de probabilidades previstas (7043,)
    grad = (X.T @ (p - y)) / m  # Gradiente da log-verossimilhança negativa média, onde (p - y) é o vetor de erros do modelo (7043,)
    return grad


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
    m = X.shape[0] # Número de clientes (7043)
    p = sigmoide(X @ w)
    d = p * (1.0 - p)  # Vetor da diagonal de D (7043,)

    # Multiplicação eficiente X^T @ D @ X sem alocar matriz (7043 x 7043)
    hess = (X.T @ (d[:, np.newaxis] * X)) / m
    return hess


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Garante que a raiz do projeto esteja no sys.path ao executar partes.py diretamente
    raiz_projeto = Path(__file__).resolve().parent.parent.parent
    if str(raiz_projeto) not in sys.path:
        sys.path.insert(0, str(raiz_projeto))

    from src.preprocessamento.processamento_dados import (
        carregar_dados_brutos,
        carregar_variaveis_json,
        preprocessar_dados,
    )

    X_df, y_s = preprocessar_dados(carregar_dados_brutos(), carregar_variaveis_json())
    X_mat = X_df.to_numpy(dtype=float)
    y_vec = y_s.to_numpy(dtype=float)

    # A gente começa com um vetor w totalmente "burro" (tudo zero)
    # Depois, fazemos um loop de repetição (o otimizador) que vai atualizando ("renovando")
    # o w passo a passo para diminuir a perda.
    w0 = np.zeros(X_mat.shape[1], dtype=float)
    f0 = funcao_objetivo(w0, X_mat, y_vec)
    g0 = gradiente(w0, X_mat, y_vec)
    H0 = hessiana(w0, X_mat, y_vec)

    print(f"f(w0)         = {f0:.6f}")
    print(f"||grad(w0)||  = {np.linalg.norm(g0):.6f}")
    print(f"Dimensão H(w0)= {H0.shape}, Número de condicionamento = {np.linalg.cond(H0):.2f}")
