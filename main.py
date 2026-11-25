"""
Ponto de entrada principal do projeto.
Executa o pré-processamento, treina os três métodos e compara os resultados.
"""
import numpy as np

from src.comparacao_sklearn.metricas import imprimir_comparacao
from src.comparacao_sklearn.modelo_sklearn import treinar_modelo_sklearn
from src.otimizacao.metodo_cauchy import otimizar_metodo_cauchy
from src.otimizacao.metodo_newton import otimizar_metodo_newton
from src.preprocessamento.processamento_dados import (
    carregar_dados_brutos,
    carregar_variaveis_json,
    preprocessar_dados,
)


def imprimir_pesos_lado_a_lado(colunas, resultados) -> None:
    """Mostra os pesos w* encontrados por cada método lado a lado."""
    cabecalho = f"{'Variável':<30}" + "".join(f" | {r.metodo:>13}" for r in resultados)
    print("\n" + cabecalho)
    print("-" * len(cabecalho))
    for i, nome in enumerate(colunas):
        valores = "".join(f" | {r.pesos[i]:+13.6f}" for r in resultados)
        print(f"{nome:<30}{valores}")


def main() -> None:
    # 1. Carrega e pré-processa os dados
    X_df, y_s = preprocessar_dados(carregar_dados_brutos(), carregar_variaveis_json())
    X = X_df.to_numpy(dtype=float)
    y = y_s.to_numpy(dtype=float)
    w0 = np.zeros(X.shape[1], dtype=float)

    # 2. Executa os três métodos de otimização
    res_cauchy = otimizar_metodo_cauchy(X, y)
    res_newton = otimizar_metodo_newton(X, y)
    res_sklearn = treinar_modelo_sklearn(X, y)
    resultados = [res_cauchy, res_newton, res_sklearn]

    # 3. Exibe métricas de desempenho e comparação dos pesos
    print(f"Dados: {X.shape[0]} clientes, {X.shape[1]} parâmetros (Churn real: {y.mean():.2%})\n")
    imprimir_comparacao(w0, resultados, X, y)
    imprimir_pesos_lado_a_lado(X_df.columns, resultados)


if __name__ == "__main__":
    main()
