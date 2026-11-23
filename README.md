# Projeto de Programação Não Linear — Previsão de *Churn* com Regressão Logística e Redução de Atributos

Este repositório contém a implementação computacional, os experimentos numéricos e o desenvolvimento do projeto para a disciplina de **Programação Não Linear (UFPR / DMAT)**.

---

## 1. Contextualização

O cancelamento inesperado de serviços por clientes (*churn*) representa um desafio relevante para empresas de telecomunicações, podendo comprometer receitas, dificultar o planejamento comercial e encarecer a aquisição de novos consumidores. Nesse contexto, técnicas de aprendizado de máquina (*Machine Learning*) e modelos estatísticos têm sido empregados na construção de modelos preditivos capazes de estimar a probabilidade de cancelamento a partir de características demográficas, informações sobre serviços contratados, duração do vínculo e dados financeiros, auxiliando na elaboração de estratégias de retenção.

Entretanto, a construção desses modelos envolve desafios relacionados à complexidade (custo) da solução, à quantidade de variáveis explicativas necessárias e à interpretabilidade dos resultados. Buscar modelos mais enxutos — que utilizem apenas um subconjunto relevante de atributos sem comprometer significativamente a capacidade preditiva — conduz à formulação de um problema de **otimização não linear irrestrita**, cuja resolução requer métodos numéricos iterativos.

---

## 2. Objetivo Geral

Investigar a aplicação de técnicas de otimização não linear na construção de modelos de regressão logística binária para a previsão de cancelamento de clientes (*churn*), utilizando uma base de dados pública do setor de telecomunicações (`telco.csv`, com $N = 7.043$ clientes e $n = 14$ parâmetros após pré-processamento).

### Objetivos Específicos

* **Redução de atributos via otimização:** Formular um problema de otimização com penalização suavizada para reduzir a quantidade de atributos do modelo, analisando o equilíbrio entre simplicidade, interpretabilidade e desempenho preditivo;
* **Implementação e comparação de métodos numéricos:** Implementar e comparar o comportamento numérico e a eficiência de convergência do **Método de Cauchy (Gradiente) com Busca Linear Exata (via Seção Áurea)** e do **Método de Newton Puro** na estimação dos parâmetros do problema proposto.

---

## 3. Estrutura da Modelagem Matemática

* **Base de dados (`data/telco.csv`):** $N = 7.043$ observações ($80\%$ conjunto de treino com $m = 5.634$ clientes e $20\%$ conjunto de teste com $1.409$ clientes) e $9$ variáveis explicativas originais ($4$ numéricas padronizadas via *Z-Score* e $5$ categóricas codificadas via *One-Hot Encoding*), resultando em $w \in \mathbb{R}^{14}$ ($w_0$ intercepto e $w_1, \dots, w_{13}$ associados aos atributos).
* **Função Objetivo (Entropia Cruzada + Penalização $\ell_1$ Suavizada):**
  $$f_\lambda(w) = f(w) + \lambda \sum_{j=1}^{13} \varphi_\delta(w_j), \quad \text{com } \varphi_\delta(s) = \sqrt{s^2 + \delta^2} - \delta$$
  em que $f(w)$ é a entropia cruzada binária média (Log-Loss), $\lambda \ge 0$ regula o equilíbrio entre ajuste e simplicidade ($\lambda = 0$ recupera o modelo completo sem penalização) e $\delta > 0$ controla a suavização diferenciável da norma $\ell_1$.
* **Métodos de Resolução:**
  1. **Método de Cauchy com Busca Linear Exata:** Direção $d^k = -\nabla f_\lambda(w^k)$ e passo $t_k = \arg\min_{t \ge 0} f_\lambda(w^k + t d^k)$ obtido pelo **Método da Seção Áurea**.
  2. **Método de Newton Puro:** Direção obtida pela resolução do sistema linear $\nabla^2 f_\lambda(w^k) d^k = -\nabla f_\lambda(w^k)$ com passo unitário ($w^{k+1} = w^k + d^k$).
