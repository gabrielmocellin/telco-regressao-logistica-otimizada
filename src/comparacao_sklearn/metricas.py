"""
Métricas simples para comparar o modelo antes e depois da otimização.

Para cada cliente i, o modelo calcula p_i = sigmoide(x_i @ w) = P(Churn = 1):
  - Probabilidade dada ao resultado correto: p_i se y_i = 1, ou (1 - p_i) se y_i = 0.
  - Acurácia: porcentagem de clientes classificados corretamente (Churn se p_i > 0.5).
"""
from typing import Dict, List
import numpy as np

from src.funcao_objetivo.partes import funcao_objetivo, sigmoide
from src.otimizacao.resultado import ResultadoOtimizacao


def avaliar_pesos(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calcula loss, probabilidade média de acerto e acurácia para um vetor de pesos w."""
    p = sigmoide(X @ w)
    prob_correta = np.where(y == 1, p, 1.0 - p)
    y_previsto = (p > 0.5).astype(float)

    perda = funcao_objetivo(w, X, y)
    return {
        "perda": perda,
        "prob_media": float(np.mean(prob_correta)),
        "prob_geometrica": float(np.exp(-perda)),
        "acuracia": float(np.mean(y_previsto == y)),
    }


def imprimir_comparacao(
    w_inicial: np.ndarray,
    resultados: List[ResultadoOtimizacao],
    X: np.ndarray,
    y: np.ndarray,
    titulo: str = "COMPARAÇÃO",
) -> None:
    """Imprime a tabela comparando o ponto inicial w0 com cada otimizador."""
    linhas = [("Início (w0 = 0)", avaliar_pesos(w_inicial, X, y), None)]
    for res in resultados:
        linhas.append((res.metodo, avaliar_pesos(res.pesos, X, y), res))

    largura = 106
    print(f"\n{titulo} ({X.shape[0]} amostras, Churn real: {y.mean():.2%})")
    print("=" * largura)
    print(
        f"{'Modelo':<18} | {'Loss':>9} | {'||Grad||':>10} | {'Prob. média':>11} | "
        f"{'e^(-Loss)':>9} | {'Acurácia':>8} | {'Iter':>6} | {'Tempo (s)':>9}"
    )
    print("-" * largura)

    for nome, m, res in linhas:
        grad_str = f"{res.norma_gradiente_final:10.2e}" if res else f"{'-':>10}"
        iter_str = f"{res.iteracoes:6d}" if res else f"{'-':>6}"
        tempo_str = f"{res.tempo_segundos:9.4f}" if res else f"{'-':>9}"
        print(
            f"{nome:<18} | {m['perda']:9.6f} | {grad_str} | {m['prob_media']:11.2%} | "
            f"{m['prob_geometrica']:9.2%} | {m['acuracia']:8.2%} | {iter_str} | {tempo_str}"
        )
    print("=" * largura)
