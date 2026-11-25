"""
Método de Newton Puro para minimização irrestrita aplicado à Regressão Logística.

Referência teórica:
  A. A. Ribeiro e E. W. Karas, "Otimização Contínua: Aspectos teóricos e computacionais".
"""
import time
from typing import List
import numpy as np

from src.funcao_objetivo.partes import funcao_objetivo, gradiente, hessiana
from src.otimizacao.resultado import ResultadoOtimizacao


def calcular_direcao_newton(H: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Resolve o sistema linear Hessiana(w_k) * d_k = -Gradiente(w_k)."""
    return np.linalg.solve(H, -grad)


def otimizar_metodo_newton(
    X: np.ndarray,
    y: np.ndarray,
    tolerancia_gradiente: float = 1e-6,
    tolerancia_passo: float = 1e-8,
    maximo_iteracoes: int = 100,
    exibir_progresso: bool = False,
) -> ResultadoOtimizacao:
    """
    Minimiza f(w) pelo Método de Newton Puro:
      1. Direção de Newton : H(w_k) * d_k = -gradiente(w_k)
      2. Atualização       : w_{k+1} = w_k + d_k  (passo t_k = 1)
    """
    w = np.zeros(X.shape[1], dtype=float) # Inicializa os pesos com zeros (modelo "burro")
    historico_perda: List[float] = []
    historico_grad: List[float] = []

    inicio = time.perf_counter()
    convergiu = False
    diff_pesos: float | None = None

    for k in range(maximo_iteracoes + 1):
        perda = funcao_objetivo(w, X, y)
        grad = gradiente(w, X, y)
        norma_g = float(np.linalg.norm(grad))

        historico_perda.append(perda)
        historico_grad.append(norma_g)

        # Se exibir_progresso for True, imprime o progresso da otimização
        if exibir_progresso:
            print(f"[Newton] iter {k:3d} | f(w) = {perda:.6f} | ||grad|| = {norma_g:.2e}")

        # Condições de parada: norma do gradiente ou tamanho do passo
        if norma_g <= tolerancia_gradiente or (diff_pesos is not None and diff_pesos <= tolerancia_passo):
            convergiu = True
            break

        if k == maximo_iteracoes:
            break
        
        H = hessiana(w, X, y)
        d = calcular_direcao_newton(H, grad)
        w_novo = w + d # Atualização dos pesos com passo t_k = 1

        diff_pesos = float(np.linalg.norm(w_novo - w)) # Calcula a diferença entre os pesos antigos e novos
        w = w_novo

    return ResultadoOtimizacao(
        metodo="Newton Puro",
        pesos=w,
        perda_final=historico_perda[-1],
        norma_gradiente_final=historico_grad[-1],
        iteracoes=len(historico_perda) - 1,
        tempo_segundos=time.perf_counter() - inicio,
        convergiu=convergiu,
        historico_perda=historico_perda,
        historico_norma_gradiente=historico_grad,
    )
