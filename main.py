from pathlib import Path

import pandas as pd


def main():
    script_dir = Path(__file__).resolve().parent
    candidatos = [
        Path.cwd() / "data" / "telco.csv",
        script_dir / "data" / "telco.csv",
    ]

    arquivo = next((caminho for caminho in candidatos if caminho.is_file()), None)

    if arquivo is None:
        raise FileNotFoundError("Não foi possível encontrar o arquivo 'telco.csv'.")

    dados_df = pd.read_csv(arquivo) # DataFrame com os dados do arquivo CSV
    print(dados_df.head())  # Exibe as primeiras linhas do DataFrame só pra verificar se os dados foram carregados corretamente


if __name__ == "__main__":
	main()
