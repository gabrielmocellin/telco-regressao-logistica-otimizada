"""
Método de Cauchy (Gradiente Descendente) com Busca Linear Exata via Seção Áurea.

Referência teórica:
  A. A. Ribeiro e E. W. Karas, "Otimização Contínua: Aspectos teóricos e computacionais".
"""
import math
import time
from typing import Callable, List, Tuple
import numpy as np

from src.funcao_objetivo.partes import funcao_objetivo, gradiente
from src.otimizacao.resultado import ResultadoOtimizacao


def encontrar_intervalo_enquadramento(
    phi: Callable[[float], float],
    passo_inicial: float = 0.5,
    maximo_expansoes: int = 30,
) -> Tuple[float, float]:
    """Fase 1 da Busca Linear: encontra um intervalo [a, b] que contém o mínimo de phi(t)."""
    a = 0.0
    s = passo_inicial
    b = 2.0 * s

    phi_s = phi(s)
    phi_b = phi(b)

    tentativas = 0
    while phi_b < phi_s and tentativas < maximo_expansoes:
        a, s, phi_s = s, b, phi_b
        b = 2.0 * b
        phi_b = phi(b)
        tentativas += 1

    return a, b


def reduzir_intervalo_secao_aurea(
    phi: Callable[[float], float],
    a: float,
    b: float,
    tolerancia_busca: float = 1e-5,
) -> float:
    """Fase 2 da Busca Linear: reduz [a, b] pela Proporção Áurea até (b - a) <= tolerância."""
    theta_1 = (3.0 - math.sqrt(5.0)) / 2.0  # ≈ 0.382
    theta_2 = 1.0 - theta_1                 # ≈ 0.618

    u = a + theta_1 * (b - a)
    v = a + theta_2 * (b - a)
    phi_u, phi_v = phi(u), phi(v)

    while (b - a) > tolerancia_busca:
        if phi_u < phi_v:
            b, v, phi_v = v, u, phi_u
            u = a + theta_1 * (b - a)
            phi_u = phi(u)
        else:
            a, u, phi_u = u, v, phi_v
            v = a + theta_2 * (b - a)
            phi_v = phi(v)

    return float(0.5 * (u + v))


def busca_linear_secao_aurea(
    w: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    tolerancia_busca: float = 1e-5,
) -> float:
    """Retorna o passo t_k > 0 que minimiza phi(t) = f(w + t * d)."""
    def phi(t: float) -> float:
        return funcao_objetivo(w + t * d, X, y)

    a, b = encontrar_intervalo_enquadramento(phi)
    return reduzir_intervalo_secao_aurea(phi, a, b, tolerancia_busca=tolerancia_busca)


def otimizar_metodo_cauchy(
    X: np.ndarray,
    y: np.ndarray,
    w_inicial: np.ndarray | None = None,
    tolerancia_gradiente: float = 1e-6,
    tolerancia_passo: float = 1e-8,
    tolerancia_busca_aurea: float = 1e-5,
    maximo_iteracoes: int = 1000,
    exibir_progresso: bool = False,
) -> ResultadoOtimizacao:
    """
    Minimiza f(w) pelo Método de Cauchy:
      1. Direção de descida : d_k = -gradiente(w_k)
      2. Busca linear exata : t_k = argmin_{t > 0} f(w_k + t * d_k)
      3. Atualização        : w_{k+1} = w_k + t_k * d_k
    """
    w = np.zeros(X.shape[1], dtype=float) if w_inicial is None else w_inicial.astype(float).copy()
    historico_perda: List[float] = []
    historico_grad: List[float] = []
    historico_t_k: List[float] = []

    inicio = time.perf_counter()
    convergiu = False
    motivo_parada = "Máximo de iterações"
    diff_pesos: float | None = None

    for k in range(maximo_iteracoes + 1):
        perda = funcao_objetivo(w, X, y)
        g = gradiente(w, X, y)
        norma_g = float(np.linalg.norm(g))

        historico_perda.append(perda)
        historico_grad.append(norma_g)

        if exibir_progresso and (k <= 5 or k % 50 == 0 or norma_g <= tolerancia_gradiente):
            print(f"[Cauchy] iter {k:3d} | f(w) = {perda:.6f} | ||grad|| = {norma_g:.2e}")

        if norma_g <= tolerancia_gradiente:
            convergiu = True
            motivo_parada = "Critério 1 (||grad||)"
            break

        if diff_pesos is not None and diff_pesos <= tolerancia_passo:
            convergiu = True
            motivo_parada = "Critério 2 (passo)"
            break

        if k == maximo_iteracoes:
            break

        d = -g
        t_k = busca_linear_secao_aurea(w, d, X, y, tolerancia_busca=tolerancia_busca_aurea)
        w_novo = w + t_k * d

        diff_pesos = float(np.linalg.norm(w_novo - w))
        historico_t_k.append(t_k)
        w = w_novo

    return ResultadoOtimizacao(
        metodo="Cauchy",
        pesos=w,
        perda_final=historico_perda[-1],
        norma_gradiente_final=historico_grad[-1],
        iteracoes=len(historico_perda) - 1,
        tempo_segundos=time.perf_counter() - inicio,
        convergiu=convergiu,
        motivo_parada=motivo_parada,
        historico_perda=historico_perda,
        historico_norma_gradiente=historico_grad,
        historico_t_k=historico_t_k,
    )
