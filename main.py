"""
Ponto de entrada principal do projeto.
Executa o pré-processamento (Treino 80% / Teste 20%) e roda os 4 experimentos numéricos.
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

    graficos = []
    graficos += experimento_1_comparacao_geral(X_treino, y_treino, X_teste, y_teste, colunas)
    graficos += [experimento_2_precisao(X_treino, y_treino)]
    graficos += [experimento_3_limite_iteracoes(X_treino, y_treino)]
    graficos += [experimento_4_pontos_iniciais(X_treino, y_treino)]

    print("\nGráficos salvos em:")
    for img in graficos:
        print(f"  - {img}")


if __name__ == "__main__":
    main()
