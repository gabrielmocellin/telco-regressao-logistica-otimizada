"""
Ponto de entrada principal do projeto.
Executa o pré-processamento (Treino 80% / Teste 20%), roda os experimentos e salva os gráficos.
"""
import numpy as np

from src.comparacao_sklearn.metricas import imprimir_comparacao
from src.comparacao_sklearn.modelo_sklearn import treinar_modelo_sklearn
from src.experimentos.graficos import (
    gerar_grafico_convergencia,
    gerar_grafico_pesos_e_desempenho,
    imprimir_tabela_pontos_iniciais,
    rodar_experimento_pontos_iniciais,
)
from src.otimizacao.metodo_cauchy import otimizar_metodo_cauchy
from src.otimizacao.metodo_newton import otimizar_metodo_newton
from src.preprocessamento.processamento_dados import (
    carregar_dados_brutos,
    carregar_variaveis_json,
    preparar_treino_teste,
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
    # 1. Carrega e divide em Treino (80%) e Teste (20%) com Z-score ajustado só no treino
    X_tr_df, X_te_df, y_tr_s, y_te_s = preparar_treino_teste(
        carregar_dados_brutos(), carregar_variaveis_json()
    )
    X_treino, X_teste = X_tr_df.to_numpy(dtype=float), X_te_df.to_numpy(dtype=float)
    y_treino, y_teste = y_tr_s.to_numpy(dtype=float), y_te_s.to_numpy(dtype=float)
    w0 = np.zeros(X_treino.shape[1], dtype=float)

    # 2. Experimento 1 & 3: treina no conjunto de treino e avalia no treino e no teste
    res_cauchy = otimizar_metodo_cauchy(X_treino, y_treino)
    res_newton = otimizar_metodo_newton(X_treino, y_treino)
    res_sklearn = treinar_modelo_sklearn(X_treino, y_treino)
    resultados = [res_cauchy, res_newton, res_sklearn]

    imprimir_comparacao(w0, resultados, X_treino, y_treino, titulo="TREINO")
    imprimir_comparacao(w0, resultados, X_teste, y_teste, titulo="TESTE")
    imprimir_pesos_lado_a_lado(X_tr_df.columns, resultados)

    # 3. Experimento 2: sensibilidade ao ponto inicial w0
    resultados_w0, img2 = rodar_experimento_pontos_iniciais(X_treino, y_treino)
    imprimir_tabela_pontos_iniciais(resultados_w0)

    # 4. Gera gráficos em docs/imgs/experimentos/
    img1 = gerar_grafico_convergencia(res_cauchy, res_newton)
    img3 = gerar_grafico_pesos_e_desempenho(
        list(X_tr_df.columns), resultados, X_treino, y_treino, X_teste, y_teste
    )
    print("\nGráficos salvos em:")
    for img in (img1, img2, img3):
        print(f"  - {img}")


if __name__ == "__main__":
    main()
