"""
Experimentos numéricos e gráficos para a Seção 3:
  1. Comparação geral (Cauchy, Newton Puro e Scikit-Learn em Treino e Teste).
  2. Influência da precisão de convergência (tolerancia_gradiente = 1e-4, 1e-6, 1e-8).
  3. Influência do limite máximo de iterações (maximo_iteracoes = 5, 50).
  4. Sensibilidade ao ponto inicial w0 (origem, aleatório moderado e distante).
"""
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.comparacao_sklearn.metricas import avaliar_pesos, imprimir_comparacao
from src.comparacao_sklearn.modelo_sklearn import treinar_modelo_sklearn
from src.otimizacao.metodo_cauchy import otimizar_metodo_cauchy
from src.otimizacao.metodo_newton import otimizar_metodo_newton
from src.otimizacao.resultado import ResultadoOtimizacao


PASTA_GRAFICOS = Path("docs") / "imgs" / "experimentos"


# =====================================================================
# FUNÇÕES AUXILIARES COMPARTILHADAS
# =====================================================================

def _salvar_figura(fig: plt.Figure, pasta_saida: Path, nome_arquivo: str) -> Path:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / nome_arquivo
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def imprimir_tabela_experimento(
    titulo: str,
    linhas: List[Tuple[str, ResultadoOtimizacao]],
) -> None:
    """Imprime tabela padronizada com os 5 indicadores de cada execução."""
    largura = 111
    print(f"\n{titulo}")
    print("=" * largura)
    print(
        f"{'Configuração':<20} | {'Método':<12} | {'Status':<10} | {'Motivo':<22} | "
        f"{'Iter':>5} | {'Loss':>10} | {'||Grad||':>10} | {'Tempo (s)':>9}"
    )
    print("-" * largura)
    for config, res in linhas:
        status = "CONVERGIU" if res.convergiu else "FALHOU"
        loss_str = f"{res.perda_final:10.6f}" if res.perda_final < 1e4 else f"{res.perda_final:10.2e}"
        print(
            f"{config:<20} | {res.metodo:<12} | {status:<10} | {res.motivo_parada:<22} | "
            f"{res.iteracoes:5d} | {loss_str} | {res.norma_gradiente_final:10.2e} | {res.tempo_segundos:9.4f}"
        )
    print("=" * largura)


def _imprimir_pesos_lado_a_lado(colunas: List[str], resultados: List[ResultadoOtimizacao]) -> None:
    cabecalho = f"{'Variável':<30}" + "".join(f" | {r.metodo:>13}" for r in resultados)
    print("\n" + cabecalho)
    print("-" * len(cabecalho))
    for i, nome in enumerate(colunas):
        valores = "".join(f" | {r.pesos[i]:+13.6f}" for r in resultados)
        print(f"{nome:<30}{valores}")


def _plotar_barras_duplas(
    ax: plt.Axes,
    categorias: List[str],
    valores_cauchy: List[float],
    valores_newton: List[float],
    titulo: str,
    ylabel: str,
    formato_label: str,
    escala_log: bool = False,
) -> None:
    """Helper para gráficos de barras agrupadas Cauchy vs Newton Puro."""
    x = np.arange(len(categorias))
    b1 = ax.bar(x - 0.18, valores_cauchy, width=0.35, label="Cauchy")
    b2 = ax.bar(x + 0.18, valores_newton, width=0.35, label="Newton Puro")
    ax.bar_label(b1, fmt=formato_label, fontsize=8, padding=2)
    ax.bar_label(b2, fmt=formato_label, fontsize=8, padding=2)
    if escala_log:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 3.0)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()


# =====================================================================
# EXPERIMENTO 1 — COMPARAÇÃO GERAL + VALIDAÇÃO COM SCIKIT-LEARN
# =====================================================================

def experimento_1_comparacao_geral(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    X_teste: np.ndarray,
    y_teste: np.ndarray,
    colunas: List[str],
    pasta_saida: Path = PASTA_GRAFICOS,
) -> List[Path]:
    """Roda Cauchy, Newton Puro e Scikit-Learn (tol=1e-6) e gera 2 gráficos."""
    w0 = np.zeros(X_treino.shape[1], dtype=float)
    res_cauchy = otimizar_metodo_cauchy(X_treino, y_treino, tolerancia_gradiente=1e-6)
    res_newton = otimizar_metodo_newton(X_treino, y_treino, tolerancia_gradiente=1e-6, maximo_iteracoes=1000)
    res_sklearn = treinar_modelo_sklearn(X_treino, y_treino, tolerancia=1e-6)
    resultados = [res_cauchy, res_newton, res_sklearn]

    imprimir_comparacao(w0, resultados, X_treino, y_treino, titulo="EXPERIMENTO 1 — TREINO")
    imprimir_comparacao(w0, resultados, X_teste, y_teste, titulo="EXPERIMENTO 1 — TESTE")
    _imprimir_pesos_lado_a_lado(colunas, resultados)

    # Gráfico 1: evolução de f(w^k) e ||grad f(w^k)||
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(res_cauchy.historico_perda, label=f"Cauchy ({res_cauchy.iteracoes} iter)", linewidth=2)
    ax1.plot(res_newton.historico_perda, "o-", label=f"Newton Puro ({res_newton.iteracoes} iter)", linewidth=2, markersize=5)
    ax1.set_title("Função Objetivo f(w^k) por Iteração")
    ax1.set_xlabel("Iteração (k)")
    ax1.set_ylabel("Perda f(w^k)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.semilogy(res_cauchy.historico_norma_gradiente, label="Cauchy", linewidth=2)
    ax2.semilogy(res_newton.historico_norma_gradiente, "o-", label="Newton Puro", linewidth=2, markersize=5)
    ax2.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="Tolerância (1e-6)")
    ax2.set_title("Norma do Gradiente ||∇f(w^k)|| (Escala Log)")
    ax2.set_xlabel("Iteração (k)")
    ax2.set_ylabel("||∇f(w^k)||")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend()
    img1 = _salvar_figura(fig1, pasta_saida, "01_exp1_convergencia.png")

    # Gráfico 2: pesos w* (Cauchy, Newton, Scikit-Learn) e acurácia Treino vs Teste
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
    y_pos = np.arange(len(colunas))
    for idx, res in enumerate(resultados):
        ax3.barh(y_pos + (idx - 1) * 0.26, res.pesos, height=0.26, label=res.metodo)
    ax3.axvline(0.0, color="black", linewidth=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(colunas, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_title("Pesos Ótimos w* por Método")
    ax3.set_xlabel("Valor do Peso")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax3.legend(fontsize=8)

    nomes = ["Início (w0=0)"] + [r.metodo for r in resultados]
    pesos_lista = [w0] + [r.pesos for r in resultados]
    acc_treino = [avaliar_pesos(w, X_treino, y_treino)["acuracia"] * 100 for w in pesos_lista]
    acc_teste = [avaliar_pesos(w, X_teste, y_teste)["acuracia"] * 100 for w in pesos_lista]

    x_pos = np.arange(len(nomes))
    b1 = ax4.bar(x_pos - 0.18, acc_treino, width=0.35, label="Treino (80%)")
    b2 = ax4.bar(x_pos + 0.18, acc_teste, width=0.35, label="Teste (20%)")
    ax4.bar_label(b1, fmt="%.1f%%", fontsize=8)
    ax4.bar_label(b2, fmt="%.1f%%", fontsize=8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(nomes, fontsize=9)
    ax4.set_ylim(60, 92)
    ax4.set_title("Acurácia: Treino vs Teste")
    ax4.set_ylabel("Acurácia (%)")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax4.legend()
    img2 = _salvar_figura(fig2, pasta_saida, "02_exp1_pesos_e_desempenho.png")

    return [img1, img2]


# =====================================================================
# EXPERIMENTO 2 — INFLUÊNCIA DA PRECISÃO (1e-4, 1e-6, 1e-8)
# =====================================================================

def experimento_2_precisao(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Path:
    """Execuções 2A, 2B e 2C variando apenas tolerancia_gradiente."""
    configs = [("2A (tol=1e-4)", 1e-4), ("2B (tol=1e-6)", 1e-6), ("2C (tol=1e-8)", 1e-8)]
    linhas: List[Tuple[str, ResultadoOtimizacao]] = []
    iter_c, iter_n, tempo_c, tempo_n = [], [], [], []

    for rotulo, tol in configs:
        res_c = otimizar_metodo_cauchy(X_treino, y_treino, tolerancia_gradiente=tol, maximo_iteracoes=1000)
        res_n = otimizar_metodo_newton(X_treino, y_treino, tolerancia_gradiente=tol, maximo_iteracoes=1000)
        linhas.extend([(rotulo, res_c), (rotulo, res_n)])
        iter_c.append(res_c.iteracoes)
        iter_n.append(res_n.iteracoes)
        tempo_c.append(res_c.tempo_segundos)
        tempo_n.append(res_n.tempo_segundos)

    imprimir_tabela_experimento("EXPERIMENTO 2 — INFLUÊNCIA DA PRECISÃO (tolerancia_gradiente)", linhas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    categorias = [c[0] for c in configs]
    _plotar_barras_duplas(ax1, categorias, iter_c, iter_n, "Iterações por Tolerância", "Iterações", "%d")
    _plotar_barras_duplas(ax2, categorias, tempo_c, tempo_n, "Tempo de Execução por Tolerância", "Tempo (s)", "%.3fs")
    return _salvar_figura(fig, pasta_saida, "03_exp2_precisao.png")


# =====================================================================
# EXPERIMENTO 3 — INFLUÊNCIA DO LIMITE MÁXIMO DE ITERAÇÕES (5 e 50)
# =====================================================================

def experimento_3_limite_iteracoes(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Path:
    """Execuções 3A e 3B com orçamento curto: maximo_iteracoes = 5 e 50."""
    limites = [("3A (max_iter=5)", 5), ("3B (max_iter=50)", 50)]
    linhas: List[Tuple[str, ResultadoOtimizacao]] = []
    loss_c, loss_n, grad_c, grad_n = [], [], [], []

    for rotulo, k_max in limites:
        res_c = otimizar_metodo_cauchy(X_treino, y_treino, maximo_iteracoes=k_max)
        res_n = otimizar_metodo_newton(X_treino, y_treino, maximo_iteracoes=k_max)
        linhas.extend([(rotulo, res_c), (rotulo, res_n)])
        loss_c.append(res_c.perda_final)
        loss_n.append(res_n.perda_final)
        grad_c.append(res_c.norma_gradiente_final)
        grad_n.append(res_n.norma_gradiente_final)

    imprimir_tabela_experimento("EXPERIMENTO 3 — LIMITE MÁXIMO DE ITERAÇÕES (5 e 50)", linhas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    categorias = [lim[0] for lim in limites]
    _plotar_barras_duplas(ax1, categorias, loss_c, loss_n, "Perda Final com Orçamento Limitado", "Loss f(w)", "%.4f")
    ax1.set_ylim(0.35, max(loss_c + loss_n) * 1.05)

    _plotar_barras_duplas(
        ax2, categorias, grad_c, grad_n,
        "Norma Final do Gradiente (Escala Log)", "||∇f(w)||", "%.1e", escala_log=True,
    )
    ax2.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="Tolerância (1e-6)")
    ax2.legend()
    return _salvar_figura(fig, pasta_saida, "04_exp3_limite_iteracoes.png")


# =====================================================================
# EXPERIMENTO 4 — SENSIBILIDADE AO PONTO INICIAL w0
# =====================================================================

def experimento_4_pontos_iniciais(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    pasta_saida: Path = PASTA_GRAFICOS,
) -> Path:
    """Testa Cauchy e Newton Puro a partir de 3 pontos iniciais w0 distintos."""
    rng = np.random.default_rng(42)
    n = X_treino.shape[1]
    pontos_iniciais: Dict[str, np.ndarray] = {
        "w0 = 0 (origem)": np.zeros(n, dtype=float),
        "w0 ~ N(0, 0.5²)": rng.normal(0.0, 0.5, size=n),
        "w0 distante (±3)": rng.choice([-3.0, 3.0], size=n),
    }

    linhas: List[Tuple[str, ResultadoOtimizacao]] = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for nome_w0, w0 in pontos_iniciais.items():
        res_c = otimizar_metodo_cauchy(X_treino, y_treino, w_inicial=w0)
        res_n = otimizar_metodo_newton(X_treino, y_treino, w_inicial=w0)
        linhas.extend([(nome_w0, res_c), (nome_w0, res_n)])

        ax1.semilogy(res_c.historico_norma_gradiente, label=f"{nome_w0} ({res_c.iteracoes} iter)", linewidth=1.8)
        status_n = f"{res_n.iteracoes} iter" if res_n.convergiu else res_n.motivo_parada
        ax2.semilogy(res_n.historico_norma_gradiente, "o-", label=f"{nome_w0} ({status_n})", linewidth=1.8, markersize=4)

    for ax, titulo in [
        (ax1, "Cauchy: ||∇f(w^k)|| por Ponto Inicial"),
        (ax2, "Newton Puro: ||∇f(w^k)|| por Ponto Inicial"),
    ]:
        ax.axhline(1e-6, color="red", linestyle="--", linewidth=1, label="Tolerância (1e-6)")
        ax.set_title(titulo)
        ax.set_xlabel("Iteração (k)")
        ax.set_ylabel("||∇f(w^k)||")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    imprimir_tabela_experimento("EXPERIMENTO 4 — SENSIBILIDADE AO PONTO INICIAL w0", linhas)
    return _salvar_figura(fig, pasta_saida, "05_exp4_pontos_iniciais.png")
