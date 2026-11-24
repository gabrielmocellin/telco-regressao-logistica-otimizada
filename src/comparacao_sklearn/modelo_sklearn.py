"""
Implementação simples e direta do treinamento da Regressão Logística
usando o scikit-learn nos mesmos dados pré-processados usados no Método de Cauchy.
detalhe: o scikit-learn não permite customizar a função objetivo, mas podemos configurar o modelo para minimizar a Log-Loss pura (sem regularização) e comparar os resultados com o nosso método.
"""
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

# Garante que a raiz do projeto esteja no sys.path mesmo ao rodar este arquivo diretamente
raiz_projeto = Path(__file__).resolve().parent.parent.parent
if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

from src.funcao_objetivo.partes import funcao_objetivo, gradiente


def treinar_modelo_sklearn(
    matriz_X: np.ndarray,
    vetor_y: np.ndarray,
    maximo_iteracoes: int = 1000,
) -> Dict[str, object]:
    """
    Treina a Regressão Logística do scikit-learn equivalente à nossa formulação:
      - C=np.inf: sem regularização (minimiza a Log-Loss pura, igual à nossa função objetivo).
      - fit_intercept=False: pois a matriz_X já possui a coluna 'Intercepto' (1.0) na posição 0 desde o pré processamento.
    """
    modelo = LogisticRegression(
        C=np.inf,
        fit_intercept=False,
        max_iter=maximo_iteracoes,
    )

    instante_inicial = time.perf_counter()
    modelo.fit(matriz_X, vetor_y)
    tempo_total = time.perf_counter() - instante_inicial

    # Extrai o vetor de pesos w* encontrado pelo sklearn (formato 1D de tamanho n)
    pesos_otimos = modelo.coef_[0].copy()

    # Avalia na nossa função objetivo e gradiente para manter a mesma base de comparação
    perda_final = funcao_objetivo(pesos_otimos, matriz_X, vetor_y)
    norma_gradiente_final = float(np.linalg.norm(gradiente(pesos_otimos, matriz_X, vetor_y)))
    numero_iteracoes = int(modelo.n_iter_[0])

    return {
        "metodo": "Scikit-Learn (LogisticRegression)",
        "modelo": modelo,
        "pesos_otimos": pesos_otimos,
        "perda_final": perda_final,
        "norma_gradiente_final": norma_gradiente_final,
        "numero_iteracoes": numero_iteracoes,
        "tempo_total_segundos": tempo_total,
    }


if __name__ == "__main__":
    from src.preprocessamento.processamento_dados import (
        carregar_dados_brutos,
        carregar_variaveis_json,
        preprocessar_dados,
    )

    X_df, y_s = preprocessar_dados(carregar_dados_brutos(), carregar_variaveis_json())
    matriz_X = X_df.to_numpy(dtype=float)
    vetor_y = y_s.to_numpy(dtype=float)

    resultado_sklearn = treinar_modelo_sklearn(
        matriz_X=matriz_X,
        vetor_y=vetor_y,
        maximo_iteracoes=1000,
    )

    print("=" * 88)
    print("RESUMO DO TREINAMENTO - SCIKIT-LEARN (LogisticRegression)")
    print("=" * 88)
    print(f"Perda final f(w*) : {resultado_sklearn['perda_final']:.8f}")
    print(f"||Gradiente(w*)|| : {resultado_sklearn['norma_gradiente_final']:.6e}")
    print(f"Iterações         : {resultado_sklearn['numero_iteracoes']}")
    print(f"Tempo total       : {resultado_sklearn['tempo_total_segundos']:.4f} segundos")
    print("=" * 88)

    print("\nPesos ótimos (w*) encontrados pelo Scikit-Learn:")
    for nome_coluna, valor_peso in zip(X_df.columns, resultado_sklearn["pesos_otimos"]):
        print(f"  {nome_coluna:<32} : {valor_peso:+.6f}")
