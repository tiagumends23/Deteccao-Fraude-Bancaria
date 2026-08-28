# Detecção de Fraudes em Transações 💳🕵️‍♂️

## Visão Geral do Projeto
Este projeto implementa um modelo de Machine Learning para detectar transações fraudulentas utilizando o famoso dataset `creditcard.csv`. O principal desafio resolvido aqui é o **desbalanceamento extremo dos dados**, onde menos de 0,2% das transações são fraudes reais. 

Modelos ingênuos que classificam todas as transações como "normais" podem atingir 99,8% de acurácia, mas falham completamente em capturar os crimes. Portanto, este projeto foge da armadilha da acurácia e foca em métricas reais de eficácia (Recall, ROC-AUC) e no ajuste fino da sensibilidade do modelo.

## Etapas da Implementação

### 1. Preparação e Limpeza dos Dados (Feature Engineering)
* **Transformação Logarítmica:** Aplicação de `np.log1p()` na variável `Amount` para suavizar e reduzir a escala de valores monetários muito altos.
* **Padronização:** Uso do `StandardScaler` do Scikit-Learn nas variáveis `Amount_log` e `Time` para garantir que as features tenham média zero e desvio padrão um.
* **Divisão Estratificada:** Utilização do `train_test_split(stratify=y)` para manter a proporção real e exata de fraudes nos conjuntos de treino e teste.

### 2. Tratamento de Dados Desbalanceados
* **SMOTE (Synthetic Minority Over-sampling Technique):** Aplicação de oversampling para criar exemplos sintéticos da classe minoritária (fraudes). Aplicado **apenas** nos dados de treino para evitar vazamento de informações (*data leakage*).

### 3. Modelagem Preditiva
* **XGBoost Classifier:** Modelo robusto baseado em árvores de decisão.
* **Pesos Balanceados:** Configuração do hiperparâmetro `scale_pos_weight` calculando a proporção exata entre transações normais e fraudes, aumentando a taxa de acerto.
* **Ajuste de Limiar (Threshold Tuning):** Mudança intencional do limiar de decisão padrão de 0.5 para **0.3**. Isso torna o sistema mais sensível a transações suspeitas, priorizando não deixar fraudes escaparem.

### 4. Avaliação e Explicabilidade
* **Métricas Reais:** Avaliação baseada em Precision, Recall e F1-Score através do `classification_report`.
* **Curva ROC e AUC:** Medição gráfica e quantitativa da capacidade do modelo de separar as classes legítimas das fraudulentas.
* **SHAP Values:** Integração com a biblioteca SHAP (SHapley Additive exPlanations) para interpretabilidade, explicando de forma visual quais variáveis tiveram mais peso na classificação de anomalias.

## Resultados Obtidos
Após o treinamento e ajuste do threshold para 0.3, o modelo obteve os seguintes resultados no conjunto de teste:

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.42      0.88      0.57        98

    accuracy                           1.00     56962
   macro avg       0.71      0.94      0.79     56962
weighted avg       1.00      1.00      1.00     56962
```
* **AUC Score:** `0.9794`

**Conclusão dos Resultados:** O modelo conseguiu atingir um excelente **Recall de 88%** para a classe de fraudes (1), o que significa que detectou a grande maioria dos crimes. Além disso, obteve um **AUC muito alto (~0.98)**, provando que é altamente capaz de separar as transações fraudulentas das legítimas.

## Tecnologias Utilizadas
* Python
* Pandas & NumPy
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* XGBoost
* SHAP
* Matplotlib & Seaborn
