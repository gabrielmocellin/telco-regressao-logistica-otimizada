"""
Implementação didática e modularizada do Método de Newton Puro para minimização
irrestrita aplicado à Regressão Logística.

Referência teórica:
  A. A. Ribeiro e E. W. Karas, "Otimização Contínua: Aspectos teóricos e computacionais".
"""
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Garante que a raiz do projeto esteja no sys.path mesmo ao rodar este arquivo diretamente
raiz_projeto = Path(__file__).resolve().parent.parent.parent
if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

from src.funcao_objetivo.partes import funcao_objetivo, gradiente, hessiana


# =====================================================================
# 1. FUNÇÃO MATEMÁTICA DA DIREÇÃO DE NEWTON
# =====================================================================

def calcular_direcao_newton(
    matriz_hessiana: np.ndarray,
    vetor_gradiente: np.ndarray,
) -> np.ndarray:
    """
    Calcula a direção de descida de Newton (d_k) resolvendo o sistema linear:
        Hessiana(w_k) * d_k = -Gradiente(w_k)

    Usamos np.linalg.solve em vez de inverter a matriz Hessiana (np.linalg.inv),
    pois resolver o sistema linear diretamente é mais rápido e numericamente estável.
    """
    direcao_newton = np.linalg.solve(matriz_hessiana, -vetor_gradiente)
    return direcao_newton


# =====================================================================
# 2. FUNÇÕES AUXILIARES DE EXIBIÇÃO (SEPARADAS DA MATEMÁTICA)
# =====================================================================

def imprimir_cabecalho_newton() -> None:
    """Imprime o cabeçalho da tabela de acompanhamento no terminal."""
    print("=" * 75)
    print("INICIANDO O MÉTODO DE NEWTON PURO")
    print("=" * 75)
    print(f"{'Iter (k)':>8} | {'Perda f(w_k)':>14} | {'||Gradiente||':>15} | {'||w_{k+1} - w_k||':>18}")
    print("-" * 75)


def imprimir_iteracao_newton(
    iteracao: int,
    perda: float,
    norma_gradiente: float,
    diferenca_pesos: float | None = None,
) -> None:
    """Imprime uma linha formatada com os dados da iteração atual."""
    passo_str = f"{diferenca_pesos:18.6e}" if diferenca_pesos is not None else f"{'-':>18}"
    print(f"{iteracao:8d} | {perda:14.8f} | {norma_gradiente:15.6e} | {passo_str}")


def imprimir_resumo_final_newton(
    convergiu: bool,
    motivo_parada: str,
    numero_iteracoes: int,
    tempo_total: float,
) -> None:
    """Imprime o quadro de resumo ao final da execução do algoritmo."""
    print("-" * 75)
    print(f"Status final : {'CONVERGIU' if convergiu else 'NÃO CONVERGIU'}")
    print(f"Motivo       : {motivo_parada}")
    print(f"Iterações    : {numero_iteracoes}")
    print(f"Tempo total  : {tempo_total:.4f} segundos")
    print("=" * 75)


# =====================================================================
# 3. ALGORITMO PRINCIPAL DO MÉTODO DE NEWTON PURO
# =====================================================================

def otimizar_metodo_newton(
    matriz_X: np.ndarray,
    vetor_y: np.ndarray,
    pesos_iniciais: np.ndarray | None = None,
    tolerancia_gradiente: float = 1e-6,
    tolerancia_passo: float = 1e-8,
    maximo_iteracoes: int = 100,
    exibir_progresso: bool = True,
) -> Dict[str, object]:
    """
    Executa o Método de Newton Puro para encontrar o vetor de pesos ótimo (w*).

    Resumo matemático de cada iteração k:
      1. Sistema linear de Newton : Hessiana(w_k) * d_k = -gradiente(w_k)
      2. Atualização (passo t_k=1): w_{k+1} = w_k + d_k
    """
    numero_parametros = matriz_X.shape[1]
    pesos_atuais = (
        np.zeros(numero_parametros, dtype=float)
        if pesos_iniciais is None
        else pesos_iniciais.astype(float).copy()
    )

    historico_perda: List[float] = []
    historico_norma_gradiente: List[float] = []
    historico_diferenca_pesos: List[float] = []
    historico_tempo_acumulado: List[float] = []

    instante_inicial = time.perf_counter()
    convergiu = False
    motivo_parada = "Atingiu o número máximo de iterações (fracasso na convergência)"

    if exibir_progresso:
        imprimir_cabecalho_newton()

    for iteracao_atual in range(maximo_iteracoes + 1):
        # 1. Avalia a função objetivo e o gradiente no ponto atual w_k
        perda_atual = funcao_objetivo(pesos_atuais, matriz_X, vetor_y)
        vetor_gradiente = gradiente(pesos_atuais, matriz_X, vetor_y)
        norma_gradiente = float(np.linalg.norm(vetor_gradiente))

        # 2. Registra métricas da iteração atual no histórico
        historico_perda.append(perda_atual)
        historico_norma_gradiente.append(norma_gradiente)
        historico_tempo_acumulado.append(time.perf_counter() - instante_inicial)

        if exibir_progresso:
            imprimir_iteracao_newton(
                iteracao=iteracao_atual,
                perda=perda_atual,
                norma_gradiente=norma_gradiente,
                diferenca_pesos=historico_diferenca_pesos[-1] if historico_diferenca_pesos else None,
            )

        # 3. Critério de parada 1 (Otimalidade): ||gradiente(w_k)|| <= 1e-6
        if norma_gradiente <= tolerancia_gradiente:
            convergiu = True
            motivo_parada = (
                f"Critério 1 atingido: ||gradiente|| ({norma_gradiente:.2e}) <= {tolerancia_gradiente:.1e}"
            )
            break

        if iteracao_atual == maximo_iteracoes:
            break

        # 4. Passo matemático de Newton: resolve H * d_k = -g e faz w_{k+1} = w_k + d_k
        matriz_hessiana = hessiana(pesos_atuais, matriz_X, vetor_y)
        direcao_newton = calcular_direcao_newton(matriz_hessiana, vetor_gradiente)
        pesos_novos = pesos_atuais + direcao_newton

        diferenca_pesos = float(np.linalg.norm(pesos_novos - pesos_atuais))
        historico_diferenca_pesos.append(diferenca_pesos)
        pesos_atuais = pesos_novos

        # 5. Critério de parada 2 (Estagnação dos pesos): ||w_{k+1} - w_k|| <= 1e-8
        if diferenca_pesos <= tolerancia_passo:
            perda_final = funcao_objetivo(pesos_atuais, matriz_X, vetor_y)
            norma_grad_final = float(np.linalg.norm(gradiente(pesos_atuais, matriz_X, vetor_y)))

            historico_perda.append(perda_final)
            historico_norma_gradiente.append(norma_grad_final)
            historico_tempo_acumulado.append(time.perf_counter() - instante_inicial)

            if exibir_progresso:
                imprimir_iteracao_newton(
                    iteracao=iteracao_atual + 1,
                    perda=perda_final,
                    norma_gradiente=norma_grad_final,
                    diferenca_pesos=diferenca_pesos,
                )

            convergiu = True
            motivo_parada = (
                f"Critério 2 atingido: ||w_{{k+1}} - w_k|| ({diferenca_pesos:.2e}) <= {tolerancia_passo:.1e}"
            )
            break

    tempo_total = time.perf_counter() - instante_inicial
    numero_iteracoes = len(historico_perda) - 1

    if exibir_progresso:
        imprimir_resumo_final_newton(convergiu, motivo_parada, numero_iteracoes, tempo_total)

    return {
        "metodo": "Método de Newton Puro",
        "pesos_otimos": pesos_atuais,
        "perda_final": historico_perda[-1],
        "norma_gradiente_final": historico_norma_gradiente[-1],
        "numero_iteracoes": numero_iteracoes,
        "convergiu": convergiu,
        "motivo_parada": motivo_parada,
        "tempo_total_segundos": tempo_total,
        "historico_perda": historico_perda,
        "historico_norma_gradiente": historico_norma_gradiente,
        "historico_diferenca_pesos": historico_diferenca_pesos,
        "historico_tempo_acumulado": historico_tempo_acumulado,
    }


if __name__ == "__main__":
    from src.preprocessamento.processamento_dados import (
        carregar_dados_brutos,
        carregar_variaveis_json,
        preprocessar_dados,
    )

    X_df, y_s = preprocessar_dados(carregar_dados_brutos(), carregar_variaveis_json())
    matriz_X = X_df.to_numpy(dtype=float)
    vetor_y = y_s.to_numpy(dtype=float)

    resultado_newton = otimizar_metodo_newton(
        matriz_X=matriz_X,
        vetor_y=vetor_y,
        tolerancia_gradiente=1e-6,
        tolerancia_passo=1e-8,
        maximo_iteracoes=100,
    )

    print("\nPesos ótimos (w*) encontrados pelo Método de Newton Puro:")
    for nome_coluna, valor_peso in zip(X_df.columns, resultado_newton["pesos_otimos"]):
        print(f"  {nome_coluna:<32} : {valor_peso:+.6f}")
