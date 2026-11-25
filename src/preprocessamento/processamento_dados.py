from pathlib import Path
from typing import Tuple
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
        return json.load(f)


def _codificar_features_e_target(
    dados_df: pd.DataFrame,
    variaveis_dict: dict,
    adicionar_intercepto: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Codifica variáveis binárias/categóricas e separa X (sem padronizar) e y."""
    variaveis_interesse = variaveis_dict.get("variaveis", {})
    if not variaveis_interesse:
        raise ValueError("A lista de variáveis de interesse está vazia no arquivo JSON.")

    numericas = variaveis_interesse["numericas"]
    binarias = variaveis_interesse["binarias"]
    categoricas = variaveis_interesse["categoricas"]
    target = variaveis_interesse["target"]

    X_num = dados_df[numericas].astype(float).copy()

    mapa_binario = {"Yes": 1.0, "No": 0.0, 1: 1.0, 0: 0.0, True: 1.0, False: 0.0}
    X_bin = pd.DataFrame(index=dados_df.index)
    for col in binarias:
        X_bin[col] = dados_df[col].map(mapa_binario).astype(float)
        if X_bin[col].isna().any():
            valores_invalidos = dados_df[col][X_bin[col].isna()].unique()
            raise ValueError(f"Valores inesperados na coluna binária '{col}': {valores_invalidos}")

    df_cat = dados_df[categoricas].fillna("None").astype(str)
    X_cat = pd.get_dummies(df_cat, columns=categoricas, drop_first=True, dtype=float)

    partes_X = [X_num, X_bin, X_cat]
    if adicionar_intercepto:
        col_intercepto = pd.DataFrame(
            {"Intercepto": np.ones(len(dados_df), dtype=float)},
            index=dados_df.index,
        )
        partes_X.insert(0, col_intercepto)

    X = pd.concat(partes_X, axis=1)

    y = dados_df[target].map(mapa_binario).astype(float)
    if y.isna().any():
        valores_invalidos = dados_df[target][y.isna()].unique()
        raise ValueError(f"Valores inesperados na coluna alvo '{target}': {valores_invalidos}")

    return X, y, numericas


def _padronizar_numericas(
    X_treino: pd.DataFrame,
    X_teste: pd.DataFrame | None,
    numericas: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """Calcula média e desvio padrão amostral (ddof=1) só no treino e aplica em ambos."""
    X_treino = X_treino.copy()
    medias = X_treino[numericas].mean(axis=0)
    desvios = X_treino[numericas].std(axis=0, ddof=1).replace(0.0, 1.0)
    X_treino[numericas] = (X_treino[numericas] - medias) / desvios

    if X_teste is not None:
        X_teste = X_teste.copy()
        X_teste[numericas] = (X_teste[numericas] - medias) / desvios

    return X_treino, X_teste


def preprocessar_dados(
    dados_df: pd.DataFrame,
    variaveis_dict: dict,
    padronizar_numericas: bool = True,
    adicionar_intercepto: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Pré-processa a base completa (sem divisão treino/teste)."""
    X, y, numericas = _codificar_features_e_target(
        dados_df, variaveis_dict, adicionar_intercepto=adicionar_intercepto
    )
    if padronizar_numericas:
        X, _ = _padronizar_numericas(X, None, numericas)
    return X, y


def preparar_treino_teste(
    dados_df: pd.DataFrame,
    variaveis_dict: dict,
    proporcao_teste: float = 0.20,
    random_state: int = 42,
    padronizar_numericas: bool = True,
    adicionar_intercepto: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide os dados em Treino (80%) e Teste (20%) de forma estratificada e
    padroniza as colunas numéricas usando apenas média e desvio do treino.
    """
    X, y, numericas = _codificar_features_e_target(
        dados_df, variaveis_dict, adicionar_intercepto=adicionar_intercepto
    )
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=random_state, stratify=y
    )
    if padronizar_numericas:
        X_treino, X_teste = _padronizar_numericas(X_treino, X_teste, numericas)
    return X_treino, X_teste, y_treino, y_teste
