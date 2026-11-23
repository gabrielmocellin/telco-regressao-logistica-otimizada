"""
Implementação didática e modularizada do Método de Cauchy (Gradiente Descendente)
com Busca Linear Exata via Seção Áurea (Proporção Áurea) aplicado à Regressão Logística.

Referência teórica e convenção de notação (passo t_k, direção d_k):
  A. A. Ribeiro e E. W. Karas, "Otimização Contínua: Aspectos teóricos e computacionais".
"""
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np

# Garante que a raiz do projeto esteja no sys.path mesmo ao rodar este arquivo diretamente
raiz_projeto = Path(__file__).resolve().parent.parent.parent
if str(raiz_projeto) not in sys.path:
    sys.path.insert(0, str(raiz_projeto))

from src.funcao_objetivo.partes import funcao_objetivo, gradiente


# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS DA BUSCA LINEAR EXATA (SEÇÃO ÁUREA)
# =====================================================================

def encontrar_intervalo_enquadramento(
    phi: Callable[[float], float],
    passo_inicial: float = 0.5,
    maximo_expansoes: int = 30,
) -> Tuple[float, float]:
    """
    Fase 1 da Busca Linear: encontra um intervalo [a, b] que contém o passo ótimo t*.
    Dobra o limite direito 'b' enquanto a função phi(t) ainda estiver descendo (phi(b) < phi(s)).
    """
    a = 0.0
    s = passo_inicial
    b = 2.0 * s

    phi_s = phi(s)
    phi_b = phi(b)

    tentativas = 0
    while phi_b < phi_s and tentativas < maximo_expansoes:
        a = s
        s = b
        phi_s = phi_b
        b = 2.0 * b
        phi_b = phi(b)
        tentativas += 1

    return a, b


def reduzir_intervalo_secao_aurea(
    phi: Callable[[float], float],
    a: float,
    b: float,
    tolerancia_busca: float = 1e-5,
) -> float:
    """
    Fase 2 da Busca Linear: reduz o intervalo [a, b] usando a Proporção Áurea
    até que a largura (b - a) seja menor ou igual à tolerância.
    """
    theta_1 = (3.0 - math.sqrt(5.0)) / 2.0  # ≈ 0.382
    theta_2 = 1.0 - theta_1                 # ≈ 0.618

    u = a + theta_1 * (b - a)
    v = a + theta_2 * (b - a)

    phi_u = phi(u)
    phi_v = phi(v)

    while (b - a) > tolerancia_busca:
        if phi_u < phi_v:
            # O mínimo está à esquerda de v (no subintervalo [a, v])
            b = v
            v = u
            phi_v = phi_u
            u = a + theta_1 * (b - a)
            phi_u = phi(u)
        else:
            # O mínimo está à direita de u (no subintervalo [u, b])
            a = u
            u = v
            phi_u = phi_v
            v = a + theta_2 * (b - a)
            phi_v = phi(v)

    # Retorna o ponto médio do pequeno intervalo final [u, v]
    t_otimo = 0.5 * (u + v)
    return float(t_otimo)


def busca_linear_secao_aurea(
    pesos_atuais: np.ndarray,
    direcao_descida: np.ndarray,
    matriz_X: np.ndarray,
    vetor_y: np.ndarray,
    tolerancia_busca: float = 1e-5,
    passo_inicial_enquadramento: float = 0.5,
) -> float:
    """
    Calcula o passo ótimo t_k > 0 que minimiza phi(t) = f(w_k + t * d_k)
    combinando a Fase 1 (enquadramento do intervalo) e a Fase 2 (Seção Áurea).
    """
    def phi(t: float) -> float:
        return funcao_objetivo(pesos_atuais + t * direcao_descida, matriz_X, vetor_y)

    a, b = encontrar_intervalo_enquadramento(phi, passo_inicial=passo_inicial_enquadramento)
    t_otimo = reduzir_intervalo_secao_aurea(phi, a, b, tolerancia_busca=tolerancia_busca)
    return t_otimo


# =====================================================================
# 2. FUNÇÕES AUXILIARES DE EXIBIÇÃO (SEPARADAS DA MATEMÁTICA)
# =====================================================================

def imprimir_cabecalho_cauchy() -> None:
    """Imprime o cabeçalho da tabela de acompanhamento no terminal."""
    print("=" * 88)
    print("INICIANDO O MÉTODO DE CAUCHY COM BUSCA LINEAR EXATA (SEÇÃO ÁUREA)")
    print("=" * 88)
    print(
        f"{'Iter (k)':>8} | {'Perda f(w_k)':>14} | {'||Gradiente||':>15} | {'Passo (t_k)':>14} | {'||w_{k+1} - w_k||':>18}"
    )
    print("-" * 88)


def imprimir_iteracao_cauchy(
    iteracao: int,
    perda: float,
    norma_gradiente: float,
    t_k: float | None = None,
    diferenca_pesos: float | None = None,
) -> None:
    """Imprime uma linha formatada com os dados da iteração atual."""
    t_k_str = f"{t_k:14.6f}" if t_k is not None else f"{'-':>14}"
    passo_str = f"{diferenca_pesos:18.6e}" if diferenca_pesos is not None else f"{'-':>18}"
    print(f"{iteracao:8d} | {perda:14.8f} | {norma_gradiente:15.6e} | {t_k_str} | {passo_str}")


def imprimir_resumo_final_cauchy(
    convergiu: bool,
    motivo_parada: str,
    numero_iteracoes: int,
    tempo_total: float,
) -> None:
    """Imprime o quadro de resumo ao final da execução do algoritmo."""
    print("-" * 88)
    print(f"Status final : {'CONVERGIU' if convergiu else 'NÃO CONVERGIU'}")
    print(f"Motivo       : {motivo_parada}")
    print(f"Iterações    : {numero_iteracoes}")
    print(f"Tempo total  : {tempo_total:.4f} segundos")
    print("=" * 88)


# =====================================================================
# 3. ALGORITMO PRINCIPAL DO MÉTODO DE CAUCHY
# =====================================================================

def otimizar_metodo_cauchy(
    matriz_X: np.ndarray,
    vetor_y: np.ndarray,
    pesos_iniciais: np.ndarray | None = None,
    tolerancia_gradiente: float = 1e-6,
    tolerancia_passo: float = 1e-8,
    tolerancia_busca_aurea: float = 1e-5,
    maximo_iteracoes: int = 1000,
    exibir_progresso: bool = True,
) -> Dict[str, object]:
    """
    Executa o Método de Cauchy (Gradiente Descendente) com Busca Linear Exata (Seção Áurea).

    Resumo matemático de cada iteração k:
      1. Direção de máxima descida : d_k = -gradiente(w_k)
      2. Busca linear exata        : t_k = argmin_{t > 0} f(w_k + t * d_k)
      3. Atualização dos pesos     : w_{k+1} = w_k + t_k * d_k
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
    historico_t_k: List[float] = []
    historico_tempo_acumulado: List[float] = []

    instante_inicial = time.perf_counter()
    convergiu = False
    motivo_parada = "Atingiu o número máximo de iterações (fracasso na convergência)"

    if exibir_progresso:
        imprimir_cabecalho_cauchy()

    for iteracao_atual in range(maximo_iteracoes + 1):
        # 1. Avalia a função objetivo e o gradiente no ponto atual w_k
        perda_atual = funcao_objetivo(pesos_atuais, matriz_X, vetor_y)
        vetor_gradiente = gradiente(pesos_atuais, matriz_X, vetor_y)
        norma_gradiente = float(np.linalg.norm(vetor_gradiente))

        # 2. Registra métricas da iteração atual no histórico
        historico_perda.append(perda_atual)
        historico_norma_gradiente.append(norma_gradiente)
        historico_tempo_acumulado.append(time.perf_counter() - instante_inicial)

        if exibir_progresso and (
            iteracao_atual <= 5
            or iteracao_atual % 25 == 0
            or norma_gradiente <= tolerancia_gradiente
            or iteracao_atual == maximo_iteracoes
        ):
            imprimir_iteracao_cauchy(
                iteracao=iteracao_atual,
                perda=perda_atual,
                norma_gradiente=norma_gradiente,
                t_k=historico_t_k[-1] if historico_t_k else None,
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

        # 4. Passo matemático de Cauchy: d_k = -gradiente(w_k) e w_{k+1} = w_k + t_k * d_k
        direcao_descida = -vetor_gradiente
        t_k = busca_linear_secao_aurea(
            pesos_atuais=pesos_atuais,
            direcao_descida=direcao_descida,
            matriz_X=matriz_X,
            vetor_y=vetor_y,
            tolerancia_busca=tolerancia_busca_aurea,
        )
        pesos_novos = pesos_atuais + t_k * direcao_descida

        diferenca_pesos = float(np.linalg.norm(pesos_novos - pesos_atuais))
        historico_t_k.append(t_k)
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
                imprimir_iteracao_cauchy(
                    iteracao=iteracao_atual + 1,
                    perda=perda_final,
                    norma_gradiente=norma_grad_final,
                    t_k=t_k,
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
        imprimir_resumo_final_cauchy(convergiu, motivo_parada, numero_iteracoes, tempo_total)

    return {
        "metodo": "Método de Cauchy com Busca Linear Exata (Seção Áurea)",
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
        "historico_t_k": historico_t_k,
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

    resultado_cauchy = otimizar_metodo_cauchy(
        matriz_X=matriz_X,
        vetor_y=vetor_y,
        tolerancia_gradiente=1e-6,
        tolerancia_passo=1e-8,
        maximo_iteracoes=1000,
    )

    print("\nPesos ótimos (w*) encontrados pelo Método de Cauchy (Seção Áurea):")
    for nome_coluna, valor_peso in zip(X_df.columns, resultado_cauchy["pesos_otimos"]):
        print(f"  {nome_coluna:<32} : {valor_peso:+.6f}")
