"""
Treinamento da Regressão Logística com scikit-learn nos mesmos dados pré-processados.
"""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.funcao_objetivo.partes import funcao_objetivo, gradiente
from src.otimizacao.resultado import ResultadoOtimizacao


def treinar_modelo_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    tolerancia: float = 1e-6,
    maximo_iteracoes: int = 1000,
) -> ResultadoOtimizacao:
    """
    Treina LogisticRegression equivalente à nossa formulação:
      - C=np.inf: sem regularização (minimiza a Log-Loss pura).
      - fit_intercept=False: a matriz X já inclui a coluna 'Intercepto' na posição 0.
      - tol=tolerancia: critério de parada alinhado com Cauchy e Newton (1e-6).
    """
    modelo = LogisticRegression(
        C=np.inf, # C=np.inf para penalty=None
        fit_intercept=False, # X já inclui a coluna de intercepto
        tol=tolerancia, # critério de parada alinhado com Cauchy e Newton
        max_iter=maximo_iteracoes, # número máximo de iterações (padrão entre os 3 métodos)
    )

    inicio = time.perf_counter()
    modelo.fit(X, y)
    tempo = time.perf_counter() - inicio

    pesos = modelo.coef_[0].copy()
    perda_final = funcao_objetivo(pesos, X, y)
    norma_grad = float(np.linalg.norm(gradiente(pesos, X, y)))
    iteracoes = int(modelo.n_iter_[0])

    return ResultadoOtimizacao(
        metodo="Scikit-Learn",
        pesos=pesos,
        perda_final=perda_final,
        norma_gradiente_final=norma_grad,
        iteracoes=iteracoes,
        tempo_segundos=tempo,
        convergiu=iteracoes < maximo_iteracoes,
    )
