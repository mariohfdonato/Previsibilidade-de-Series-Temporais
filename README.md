# Previsibilidade de Séries Temporais (Bolsa de Valores)

Fala, pessoal! Este repositório tem como objetivo manter os dados e scripts necessários para funcionar o meu Dashboard feito com a aplicação Streamlit do link: https://previsibilidade-de-series-temporias-dadvy5ywqvxxt8qct2vhip.streamlit.app/

De modo mais específico, nesse Dashboard interativo eu coloco alguns resultados que eu obtive a partir de um modelo de Deep Learning que fiz do zero para essencialmente tentar prever o preço de diferentes ações no decorrer do tempo. 
Nesse modelo, tenho por input as séries temporais passadas (período de um mês, por exemplo) de um conjunto escolhido de ações. Esses dados, com os preços de abertura e fechamento, são obtidos de maneira automática a partir da API do Yahoo Finance a partir de um script em Python.

O output típico é um conjunto de gráficos que mostram as probabilidades das ações terem determinado preço em cada instante de tempo num período futuro específico (uma semana ou uma quinzena, por exemplo).

<img src="/imagens/PETR4(mês).png">

Com esses gráficos, é possível determinar três curvas: uma curva com valor médio esperado do preço, e duas outras curvas (uma otimista e outra pessimista) calculadas somando e subtraindo três vezes o valor de um desvio padrão para cada instante de tempo.

<img src="/imagens/newplot.png">

A partir dessas curvas, podemos estimar uma região provável de preço que as ações vão se manter no período de tempo futuro.

P.S.: Os resultados presentes no Dashboard foram calculados no dia 22/03/2024, com os dados das séries temporais passadas terminando nos valores de fechamento do dia 21/03/2024
