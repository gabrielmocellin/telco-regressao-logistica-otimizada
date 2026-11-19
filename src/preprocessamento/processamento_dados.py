from pathlib import Path
import pandas as pd
import json

def carregar_dados_brutos(nome_csv: str = "telco.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    candidatos = [
        Path.cwd() / "data" / nome_csv,
        script_dir / "data" / nome_csv,
    ]

    arquivo = next((caminho for caminho in candidatos if caminho.is_file()), None)

    if arquivo is None:
        raise FileNotFoundError("Não foi possível encontrar o arquivo 'telco.csv'.")

    dados_df = pd.read_csv(arquivo) # DataFrame com os dados do arquivo CSV
    # print(dados_df.head())  # Exibe as primeiras linhas do DataFrame só pra verificar se os dados foram carregados corretamente
    
    return dados_df

def carregar_variaveis_json(nome_json: str = "variaveis.json") -> dict:
    script_dir = Path(__file__).resolve().parent
    candidatos = [
        Path.cwd() / nome_json,
        script_dir / nome_json,
    ]

    arquivo = next((caminho for caminho in candidatos if caminho.is_file()), None)

    if arquivo is None:
        raise FileNotFoundError("Não foi possível encontrar o arquivo 'variaveis.json'.")

    with open(arquivo, 'r') as f:
        variaveis_dict = json.load(f)
    
    return variaveis_dict

def preprocessar_dados(dados_df: pd.DataFrame, variaveis_dict: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pré-processa os dados do DataFrame, selecionando apenas as variáveis de interesse
    e preparando a matriz de atributos X e o vetor resposta y.
    """
    
    variaveis_interesse = variaveis_dict.get("variaveis", [])

    target = variaveis_interesse["target"]

    features = (
        variaveis_interesse["numericas"]
        + variaveis_interesse["binarias"]
        + variaveis_interesse["categoricas"]
    )
    todas_colunas = features + [target]
    
    if not variaveis_interesse:
        raise ValueError("A lista de variáveis de interesse está vazia no arquivo JSON.")

    # Seleciona apenas as colunas de interesse
    dados_selecionados = dados_df[todas_colunas]

    # Supondo que a última coluna seja a variável resposta (target)
    X = dados_selecionados[features]  # Todas as colunas exceto a última
    y = dados_selecionados[target]   # Apenas a última coluna

    return X, y

if __name__ == "__main__":
    dados_df = carregar_dados_brutos()
    variaveis_dict = carregar_variaveis_json()
    X, y = preprocessar_dados(dados_df, variaveis_dict)
    print(X.head())  # Exibe as primeiras linhas da matriz de atributos X
    print(y.head())  # Exibe as primeiras linhas do vetor resposta y