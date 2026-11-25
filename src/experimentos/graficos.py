"""
Experimentos numéricos e geração de gráficos para a Seção 3 do relatório:
  1. Convergência: Método de Cauchy vs Método de Newton Puro (perda e ||grad||).
  2. Sensibilidade ao ponto inicial w0 (origem, pequeno aleatório e distante).
  3. Desempenho preditivo (Treino vs Teste) e comparação dos pesos com Scikit-Learn.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.comparacao_sklearn.metricas import avaliar_pesos
from src.otimizacao.metodo_cauchy import otimizar_metodo_cauchy
from src.otimizacao.metodo_newton import otimizar_metodo_newton
from src.otimizacao.resultado import ResultadoOtimizacao


PASTA_GRAFICOS = Path("docs") / "imgs" / "experimentos"


def gerar_grafico_convergencia(
    res_cauchy: ResultadoOtimizacao,
    res_newton: ResultadoOtimizacao,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Path:
    """Gráfico 1: evolução de f(w^k) e ||grad f(w^k)|| (escala log) por iteração."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Perda f(w^k)
    ax1.plot(res_cauchy.historico_perda, label=f"Cauchy ({res_cauchy.iteracoes} iter)", linewidth=2)
    ax1.plot(res_newton.historico_perda, "o-", label=f"Newton Puro ({res_newton.iteracoes} iter)", linewidth=2, markersize=5)
    ax1.set_title("Função Objetivo f(w^k) por Iteração")
    ax1.set_xlabel("Iteração (k)")
    ax1.set_ylabel("Perda f(w^k)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # Subplot 2: Norma do gradiente em escala log
    ax2.semilogy(res_cauchy.historico_norma_gradiente, label="Cauchy", linewidth=2)
    ax2.semilogy(res_newton.historico_norma_gradiente, "o-", label="Newton Puro", linewidth=2, markersize=5)
    ax2.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="Tolerância (1e-6)")
    ax2.set_title("Norma do Gradiente ||∇f(w^k)|| (Escala Log)")
    ax2.set_xlabel("Iteração (k)")
    ax2.set_ylabel("||∇f(w^k)||")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    caminho = pasta_saida / "01_convergencia_cauchy_vs_newton.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def rodar_experimento_pontos_iniciais(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Tuple[List[Tuple[str, ResultadoOtimizacao]], Path]:
    """Experimento 2: testa Cauchy e Newton a partir de 3 pontos iniciais diferentes."""
    rng = np.random.default_rng(42)
    n = X_treino.shape[1]

    pontos_iniciais: Dict[str, np.ndarray] = {
        "w0 = 0 (origem)": np.zeros(n, dtype=float),
        "w0 ~ N(0, 0.5²)": rng.normal(0.0, 0.5, size=n),
        "w0 distante (±3)": rng.choice([-3.0, 3.0], size=n),
    }

    resultados_exp: List[Tuple[str, ResultadoOtimizacao]] = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for nome_w0, w0 in pontos_iniciais.items():
        res_c = otimizar_metodo_cauchy(X_treino, y_treino, w_inicial=w0)
        res_n = otimizar_metodo_newton(X_treino, y_treino, w_inicial=w0)
        resultados_exp.append((nome_w0, res_c))
        resultados_exp.append((nome_w0, res_n))

        ax1.semilogy(res_c.historico_norma_gradiente, label=f"{nome_w0} ({res_c.iteracoes} iter)", linewidth=1.8)
        status_n = f"{res_n.iteracoes} iter" if res_n.convergiu else res_n.motivo_parada
        ax2.semilogy(res_n.historico_norma_gradiente, "o-", label=f"{nome_w0} ({status_n})", linewidth=1.8, markersize=4)

    for ax, titulo in [(ax1, "Cauchy: ||∇f(w^k)|| por Ponto Inicial"), (ax2, "Newton Puro: ||∇f(w^k)|| por Ponto Inicial")]:
        ax.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="Tolerância (1e-6)")
        ax.set_title(titulo)
        ax.set_xlabel("Iteração (k)")
        ax.set_ylabel("||∇f(w^k)||")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / "02_sensibilidade_ponto_inicial.png"
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return resultados_exp, caminho


def imprimir_tabela_pontos_iniciais(resultados_exp: List[Tuple[str, ResultadoOtimizacao]]) -> None:
    """Imprime o resumo do experimento com diferentes pontos iniciais w0."""
    largura = 98
    print("\nEXPERIMENTO 2 — SENSIBILIDADE AO PONTO INICIAL w0")
    print("=" * largura)
    print(f"{'Ponto Inicial':<18} | {'Método':<12} | {'Status':<11} | {'Motivo':<22} | {'Iter':>5} | {'Loss':>9} | {'Tempo (s)':>9}")
    print("-" * largura)
    for nome_w0, res in resultados_exp:
        status = "CONVERGIU" if res.convergiu else "FALHOU"
        print(
            f"{nome_w0:<18} | {res.metodo:<12} | {status:<11} | "
            f"{res.motivo_parada:<22} | {res.iteracoes:5d} | {res.perda_final:9.6f} | {res.tempo_segundos:9.4f}"
        )
    print("=" * largura)


def gerar_grafico_pesos_e_desempenho(
    colunas: List[str],
    resultados: List[ResultadoOtimizacao],
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_teste: np.ndarray,
    y_teste: np.ndarray,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Path:
    """Gráfico 3: pesos finais w* dos métodos e acurácia Treino vs Teste."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Pesos w* encontrados (Cauchy, Newton Puro, Scikit-Learn)
    y_pos = np.arange(len(colunas))
    largura_barra = 0.26
    for idx, res in enumerate(resultados):
        ax1.barh(y_pos + (idx - 1) * largura_barra, res.pesos, height=largura_barra, label=res.metodo)
    ax1.axvline(0.0, color="black", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(colunas, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title("Pesos Ótimos w* por Método")
    ax1.set_xlabel("Valor do Peso")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # Subplot 2: Acurácia no Treino vs Teste
    w0 = np.zeros(X_treino.shape[1], dtype=float)
    nomes = ["Início (w0=0)"] + [r.metodo for r in resultados]
    pesos_lista = [w0] + [r.pesos for r in resultados]

    acc_treino = [avaliar_pesos(w, X_treino, y_treino)["acuracia"] * 100 for w in pesos_lista]
    acc_teste = [avaliar_pesos(w, X_teste, y_teste)["acuracia"] * 100 for w in pesos_lista]

    x_pos = np.arange(len(nomes))
    b1 = ax2.bar(x_pos - 0.18, acc_treino, width=0.35, label="Treino (80%)")
    b2 = ax2.bar(x_pos + 0.18, acc_teste, width=0.35, label="Teste (20%)")
    ax2.bar_label(b1, fmt="%.1f%%", fontsize=8)
    ax2.bar_label(b2, fmt="%.1f%%", fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(nomes, fontsize=9)
    ax2.set_ylim(60, 92)
    ax2.set_title("Acurácia: Treino vs Teste")
    ax2.set_ylabel("Acurácia (%)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    caminho = pasta_saida / "03_pesos_e_desempenho_treino_teste.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho
