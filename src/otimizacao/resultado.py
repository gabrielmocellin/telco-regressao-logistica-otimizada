"""
Estrutura padronizada para guardar o resultado de cada método de otimização.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ResultadoOtimizacao:
    """Armazena os pesos finais, métricas de convergência e histórico de iterações."""

    metodo: str
    pesos: np.ndarray
    perda_final: float
    norma_gradiente_final: float
    iteracoes: int
    tempo_segundos: float
    convergiu: bool = True
    historico_perda: List[float] = field(default_factory=list)
    historico_norma_gradiente: List[float] = field(default_factory=list)
    historico_t_k: List[float] = field(default_factory=list)
