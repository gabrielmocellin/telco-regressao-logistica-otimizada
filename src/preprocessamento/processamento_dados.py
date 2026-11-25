from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd


def carregar_dados_brutos(nome_csv: str = "telco.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    candidatos = [
        Path.cwd() / "data" / nome_csv,
        script_dir / "data" / nome_csv,
        script_dir.parent.parent / "data" / nome_csv,
    ]

    arquivo = next((caminho for caminho in candidatos if caminho.is_file()), None)

    if arquivo is None:
        raise FileNotFoundError(f"Não foi possível encontrar o arquivo '{nome_csv}'.")

    # keep_default_na=False evita que a categoria "None" (de Internet Type) vire NaN
    # se virar NaN, o One-Hot Encoding vai gerar uma coluna a menos, e o modelo vai falhar
    dados_df = pd.read_csv(arquivo, keep_default_na=False)
    return dados_df


def carregar_variaveis_json(nome_json: str = "variaveis.json") -> dict:
    script_dir = Path(__file__).resolve().parent
    candidatos = [
        Path.cwd() / nome_json,
        Path.cwd() / "src" / "preprocessamento" / nome_json,
        script_dir / nome_json,
    ]

    arquivo = next((caminho for caminho in candidatos if caminho.is_file()), None)

    if arquivo is None:
        raise FileNotFoundError(f"Não foi possível encontrar o arquivo '{nome_json}'.")

    with open(arquivo, "r", encoding="utf-8") as f:
        variaveis_dict = json.load(f)

    return variaveis_dict


def preprocessar_dados(
    dados_df: pd.DataFrame,
    variaveis_dict: dict,
    padronizar_numericas: bool = True,
    adicionar_intercepto: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pré-processa os dados do DataFrame, selecionando apenas as variáveis de interesse
    e transformando todas as colunas em valores numéricos para a matriz X e o vetor y:
      1. Numéricas: convertidas para float e padronizadas (Z-score).
      2. Binárias ("Yes"/"No"): mapeadas para 1.0 e 0.0.
      3. Categóricas (k categorias): codificadas via One-Hot Encoding com k-1 colunas (drop_first=True).
      4. Intercepto: coluna de 1.0 inserida na primeira posição (índice 0) de X.
      5. Target ("Churn Label"): mapeado de "Yes"/"No" para 1.0 e 0.0.
    """
    variaveis_interesse = variaveis_dict.get("variaveis", {})
    if not variaveis_interesse:
        raise ValueError("A lista de variáveis de interesse está vazia no arquivo JSON.")

    numericas = variaveis_interesse["numericas"]
    binarias = variaveis_interesse["binarias"]
    categoricas = variaveis_interesse["categoricas"]
    target = variaveis_interesse["target"]

    # 1. Processamento das variáveis numéricas
    X_num = dados_df[numericas].astype(float).copy()
    if padronizar_numericas: # Padronização Z-score: (x - média) / desvio padrão
        medias = X_num.mean(axis=0) # 0 axis refere-se às colunas (variáveis), 1 axis refere-se às linhas (amostras)
        desvios = X_num.std(axis=0, ddof=0).replace(0.0, 1.0) # Evita divisão por zero se o desvio padrão for zero (quando todos os valores forem iguais)
        X_num = (X_num - medias) / desvios

    # 2. Processamento das variáveis binárias ("Yes" -> 1.0, "No" -> 0.0)
    mapa_binario = {"Yes": 1.0, "No": 0.0, 1: 1.0, 0: 0.0, True: 1.0, False: 0.0}
    X_bin = pd.DataFrame(index=dados_df.index)
    for col in binarias:
        X_bin[col] = dados_df[col].map(mapa_binario).astype(float)
        if X_bin[col].isna().any():
            valores_invalidos = dados_df[col][X_bin[col].isna()].unique()
            raise ValueError(f"Valores inesperados na coluna binária '{col}': {valores_invalidos}")

    # 3. Processamento das variáveis categóricas (One-Hot Encoding com k - 1 colunas)
    df_cat = dados_df[categoricas].fillna("None").astype(str)
    X_cat = pd.get_dummies(
        df_cat,
        columns=categoricas,
        drop_first=True,
        dtype=float,
    )

    # 4. Concatenação da matriz de atributos X
    partes_X = [X_num, X_bin, X_cat]
    if adicionar_intercepto:
        col_intercepto = pd.DataFrame(
            {"Intercepto": np.ones(len(dados_df), dtype=float)},
            index=dados_df.index,
        )
        partes_X.insert(0, col_intercepto)

    X = pd.concat(partes_X, axis=1)

    # 5. Processamento da variável resposta (target: "Yes" -> 1.0, "No" -> 0.0)
    y = dados_df[target].map(mapa_binario).astype(float)
    if y.isna().any():
        valores_invalidos = dados_df[target][y.isna()].unique()
        raise ValueError(f"Valores inesperados na coluna alvo '{target}': {valores_invalidos}")

    return X, y
