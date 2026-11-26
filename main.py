"""
Ponto de entrada principal do projeto.
Executa o pré-processamento (Treino 80% / Teste 20%), roda os 4 experimentos numéricos
e gera os gráficos estáticos, as animações de convergência (.gif e .mp4) e os CSVs de resultados.
"""
from src.experimentos.experimentos import (
    experimento_1_comparacao_geral,
    experimento_2_precisao,
    experimento_3_limite_iteracoes,
    experimento_4_pontos_iniciais,
)
from src.preprocessamento.processamento_dados import (
    carregar_dados_brutos,
    carregar_variaveis_json,
    preparar_treino_teste,
)


def main() -> None:
    X_tr_df, X_te_df, y_tr_s, y_te_s = preparar_treino_teste(
        carregar_dados_brutos(), carregar_variaveis_json()
    )
    X_treino, X_teste = X_tr_df.to_numpy(dtype=float), X_te_df.to_numpy(dtype=float)
    y_treino, y_teste = y_tr_s.to_numpy(dtype=float), y_te_s.to_numpy(dtype=float)
    colunas = list(X_tr_df.columns)

    arquivos_gerados = []
    arquivos_gerados += experimento_1_comparacao_geral(
        X_treino, y_treino, X_teste, y_teste, colunas, gerar_animacao=True
    )
    arquivos_gerados += experimento_2_precisao(X_treino, y_treino)
    arquivos_gerados += experimento_3_limite_iteracoes(X_treino, y_treino)
    arquivos_gerados += experimento_4_pontos_iniciais(X_treino, y_treino)

    midias = [a for a in arquivos_gerados if a.suffix != ".csv"]
    tabelas_csv = [a for a in arquivos_gerados if a.suffix == ".csv"]

    print("\nGráficos e animações salvos em:")
    for arq in midias:
        print(f"  - {arq}")

    print("\nResultados em CSV salvos em:")
    for csv in tabelas_csv:
        print(f"  - {csv}")


if __name__ == "__main__":
    main()
