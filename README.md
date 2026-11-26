# Projeto de Programação Não Linear — Previsão de *Churn* com Regressão Logística

Este repositório contém a implementação computacional, os experimentos numéricos e o desenvolvimento do projeto para a disciplina de **Programação Não Linear (UFPR / DMAT)**.

---

## 1. Contextualização

O cancelamento inesperado de serviços por clientes (*churn*) representa um desafio relevante para empresas de telecomunicações, podendo comprometer receitas, dificultar o planejamento comercial e encarecer a aquisição de novos consumidores. Nesse contexto, técnicas de aprendizado de máquina (*Machine Learning*) e modelos estatísticos têm sido empregados na construção de modelos preditivos capazes de estimar a probabilidade de cancelamento a partir de características demográficas, informações sobre serviços contratados, duração do vínculo e dados financeiros, auxiliando na elaboração de estratégias de retenção.

Para traduzir esse objetivo em um problema tratável numericamente, modela-se a probabilidade de cancelamento de um cliente por meio da função logística (sigmoide), que mapeia a combinação linear das características do consumidor para um valor contínuo em $(0, 1)$. A estimação dos coeficientes (pesos) por máxima verossimilhança conduz à minimização da função de perda de entropia cruzada binária (*Log-Loss*) — um problema clássico de **otimização não linear irrestrita**, cuja condição de otimalidade $\nabla f(w) = 0$ forma um sistema não linear sem solução analítica fechada, exigindo métodos numéricos iterativos de primeira e segunda ordem.

---

## 2. Objetivo Geral

Investigar e aplicar técnicas de otimização não linear irrestrita na estimação dos parâmetros de um modelo de regressão logística binária para a previsão de cancelamento de clientes (*churn*), utilizando uma base de dados pública do setor de telecomunicações ([data/telco.csv](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/data/telco.csv), com $N = 7.043$ clientes e $n = 14$ parâmetros após pré-processamento).

### Objetivos Específicos

* **Modelagem e pré-processamento sem vazamento de dados (*data leakage*):** Selecionar atributos relevantes, remover variáveis que carregam informação direta do desfecho, realizar divisão estratificada em conjuntos de treino ($80\%$) e teste ($20\%$), codificar variáveis categóricas (*One-Hot Encoding*) e padronizar variáveis numéricas via *Z-Score* ajustado exclusivamente sobre o treino;
* **Implementação e comparação de métodos de 1ª e 2ª ordem:** Implementar do zero o **Método de Cauchy (Gradiente Descendente) com Busca Linear Exata (via Enquadramento e Seção Áurea)** e o **Método de Newton Puro**, comparando a velocidade de convergência, o número de iterações, o custo computacional e a evolução de $f(w^k)$ e $\|\nabla f(w^k)\|_2$;
* **Análise de sensibilidade ao ponto inicial ($w^0$):** Investigar o comportamento de convergência global do Método de Cauchy frente à convergência local quadrática (e possível instabilidade longe da solução) do Método de Newton Puro a partir de diferentes inicializações ($w^0 = 0$, $w^0 \sim \mathcal{N}(0, 0{,}5^2)$ e $w^0$ distante em $\{-3, +3\}^n$);
* **Validação e avaliação de desempenho preditivo:** Validar os pesos ótimos $w^*$ obtidos pelos métodos implementados contra a solução de referência do `scikit-learn` (`LogisticRegression` sem penalização) e avaliar a capacidade de generalização do modelo nos conjuntos de treino e teste.

---

## 3. Estrutura da Modelagem Matemática

* **Base de dados ([data/telco.csv](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/data/telco.csv)) e Pré-processamento ([src/preprocessamento/](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/preprocessamento/processamento_dados.py)):**
  * $N = 7.043$ observações divididas de forma estratificada pela variável alvo `Churn Label` em $80\%$ para treino ($m = 5.634$ clientes) e $20\%$ para teste ($1.409$ clientes).
  * Seleção de $9$ variáveis explicativas originais definidas em [variaveis.json](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/preprocessamento/variaveis.json) (excluindo variáveis com vazamento direto como `Churn Score`, `Churn Category`, `Churn Reason`, `Customer Status` e `Satisfaction Score`):
    * **4 numéricas** padronizadas via *Z-Score* ($z_{ij} = (x_{ij} - \bar{x}_j) / s_j$ com média e desvio amostral calculados apenas no treino): `Age`, `Tenure in Months`, `Monthly Charge`, `Number of Referrals`;
    * **2 binárias** mapeadas para $\{0, 1\}$: `Paperless Billing`, `Married`;
    * **3 categóricas** codificadas via *One-Hot Encoding* (`drop_first=True`, gerando $7$ variáveis indicadoras): `Contract` (*One Year*, *Two Year*), `Internet Type` (*DSL*, *Fiber Optic*, *None*) e `Payment Method` (*Credit Card*, *Mailed Check*).
  * Acrescentando a coluna unitária de `Intercepto`, cada cliente $i$ é representado por $x_i \in \mathbb{R}^{14}$ e o vetor de decisão é $w = (w_0, w_1, \dots, w_{13})^\top \in \mathbb{R}^{14}$.

* **Função Objetivo e Derivadas ([src/funcao_objetivo/partes.py](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/funcao_objetivo/partes.py)):**
  * **Probabilidade estimada (Sigmoide):**
    $$\hat{p}_i(w) = \sigma(x_i^\top w) = \frac{1}{1 + e^{-x_i^\top w}}$$
  * **Função Objetivo (Entropia Cruzada Binária / Log-Verossimilhança Negativa Média):**
    $$\min_{w \in \mathbb{R}^{14}} f(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \ln \hat{p}_i(w) + (1 - y_i) \ln(1 - \hat{p}_i(w)) \right] = \frac{1}{m} \sum_{i=1}^{m} \left[ \ln(1 + e^{x_i^\top w}) - y_i x_i^\top w \right]$$
  * **Gradiente e Hessiana:**
    $$\nabla f(w) = \frac{1}{m} X^\top (\hat{p}(w) - y), \qquad \nabla^2 f(w) = \frac{1}{m} X^\top D(w) X$$
    em que $D(w) = \mathrm{diag}\big(\hat{p}_i(w)(1 - \hat{p}_i(w))\big)$.

* **Métodos de Resolução ([src/otimizacao/](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/otimizacao/)):**
  1. **Método de Cauchy com Busca Linear Exata ([metodo_cauchy.py](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/otimizacao/metodo_cauchy.py)):** Direção de máxima descida $d^k = -\nabla f(w^k)$ e passo ótimo $t_k = \arg\min_{t > 0} f(w^k + t d^k)$ determinado em duas fases: enquadramento do intervalo $[a, b]$ seguido de redução pelo **Método da Seção Áurea** (tolerância $10^{-5}$).
  2. **Método de Newton Puro ([metodo_newton.py](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/src/otimizacao/metodo_newton.py)):** Direção de segunda ordem obtida pela resolução do sistema linear $\nabla^2 f(w^k) d^k = -\nabla f(w^k)$ com passo unitário ($w^{k+1} = w^k + d^k$), monitorando singularidade da Hessiana ou divergência numérica.
  * **Critérios de Parada:** $\|\nabla f(w^k)\|_2 \le 10^{-6}$ (Critério 1), $\|w^{k+1} - w^k\|_2 \le 10^{-8}$ (Critério 2) ou limite máximo de iterações ($1.000$ para Cauchy e $100$ para Newton).

---

## 4. Experimentos Numéricos e Estrutura do Código

O ponto de entrada [main.py](file:///c:/Users/gabri/OneDrive/Documentos/dev/1.Materias_Faculdade/1.Otimizacao_continua/telco-regressao-logistica-otimizada/main.py) executa o fluxo completo de pré-processamento, otimização e avaliação, gerando tabelas comparativas no terminal e salvando as figuras em `docs/imgs/experimentos/`:

1. **Convergência Cauchy vs. Newton (`01_convergencia_cauchy_vs_newton.png`):** Evolução da função objetivo $f(w^k)$ e da norma do gradiente $\|\nabla f(w^k)\|_2$ (em escala logarítmica) por iteração a partir da origem $w^0 = 0$.
2. **Sensibilidade ao Ponto Inicial (`02_sensibilidade_ponto_inicial.png`):** Comparação da robustez dos dois métodos partindo de $w^0 = 0$, $w^0 \sim \mathcal{N}(0, 0{,}5^2)$ e $w^0 \in \{-3, +3\}^{14}$.
3. **Validação dos Pesos e Desempenho Preditivo (`03_pesos_e_desempenho_treino_teste.png`):** Comparação dos pesos ótimos $w^*$ encontrados por Cauchy, Newton Puro e `scikit-learn`, juntamente com as métricas de *Log-Loss*, probabilidade média atribuída ao rótulo correto, $e^{-\text{Loss}}$ e acurácia nos conjuntos de treino e teste.

