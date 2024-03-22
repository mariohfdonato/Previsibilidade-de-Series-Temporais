import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from datetime import datetime

amplitude = 5.0
resolucao = 65
n_particoes = (resolucao - 1)//2

st.set_page_config(layout="wide")

end_data = '2024-03-22'
end_data1 = '22-03-2024'

dados_varias = pd.read_csv(f'Cotação da Carteira último mês - ({end_data}).csv')
#dados_varias.drop(['Unnamed: 0'], axis = 1, inplace = True)
dados_varias.index = list(range(-40,0))
dados_varias_2m = pd.read_csv(f'Cotação da Carteira 2m - ({end_data}).csv')
dados_varias_4m = pd.read_csv(f'Cotação da Carteira 4m - ({end_data}).csv')
dados_varias_6m = pd.read_csv(f'Cotação da Carteira 6m - ({end_data}).csv')

dicionario = {}
for i in range(0,len(list(dados_varias.columns))):
    dicionario[list(dados_varias.columns)[i]] = i
    
previsao_6m = np.load(f'PrevisãoCarteira(T6m) - {end_data}.npy')
previsao_4m = np.load(f'PrevisãoCarteira(T4m) - {end_data}.npy')
previsao_2m = np.load(f'PrevisãoCarteira(T2m) - {end_data}.npy')  

st.title(f'Resultados de previsibilidade da Carteira de Ações para o próximo mês (dia {end_data1})')
#st.caption('Lembre-se de rodar o modelo de DeepLearning ao menos uma vez antes de rodar esta aplicação.')
st.write('Aqui estão os resultados do meu modelo de Deep Learning com a previsão do valor de 20 ativos negocioados na Bolsa de Valores Brasileira. Os valores diários de abertura e fechamento dos ativos foram coletados da base de dados do Yahoo Finance.')

acao = st.selectbox('Selecione o ativo da Carteira', list(dados_varias.columns))

aba_probabilidade, aba_margem, aba_sobreoautor = st.tabs(['Análise probabilística', 'Análise de média e intervalo de confiança', 'Sobre o autor'])

with aba_probabilidade: 
    col1, col2, col3 = st.columns(3)


    with col1:
        #ultimo mes
        fig_ultimo_mes = px.line(dados_varias[acao] , range_x=[-40,0] ,width = 1176, height=445, title = f'Variação de {acao} no último mês (tempo zero é o presente)')
        fig_ultimo_mes.update_layout(xaxis_title = 'Tempo (dt = 2 equivale a um dia)')
        fig_ultimo_mes.update_layout(yaxis_title = 'Valor (R$)')
        st.plotly_chart(fig_ultimo_mes, use_container_width = True)

        #previsão 6m
        df_acao = pd.DataFrame(np.transpose(previsao_6m[dicionario[acao],:,:])).iloc[::-1]
        df_acao['indice_bruto'] = list(df_acao.index)
        df_acao['valor_normalizado'] = amplitude*(df_acao['indice_bruto'] - n_particoes)/(n_particoes)
        df_acao['valor_real'] = (df_acao['valor_normalizado']*dados_varias_6m[acao].std() + dados_varias_6m[acao].mean()).round(3)

        df_acao.index = list(df_acao['valor_real'])
        df_acao.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)



        fig_6m, ax_6m = plt.subplots( figsize = (10,5))
        #plt.figure(figsize=(15,5))
        ax_6m = sns.heatmap(df_acao, cbar_kws={'label': 'Frequência normalizada'})
        ax_6m.set_title('Previsão de '+(acao[:-3])+' para o mês (treino 6 meses)')
        ax_6m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        ax_6m.set_ylabel('Valor em R$');
        st.pyplot(fig_6m, use_container_width = True)


    with col2:
        # previsao 4m
        df_acao2 = pd.DataFrame(np.transpose(previsao_4m[dicionario[acao],:,:])).iloc[::-1]
        df_acao2['indice_bruto'] = list(df_acao2.index)
        df_acao2['valor_normalizado'] = amplitude*(df_acao2['indice_bruto'] - n_particoes)/(n_particoes)
        df_acao2['valor_real'] = (df_acao2['valor_normalizado']*dados_varias_4m[acao].std() + dados_varias_4m[acao].mean()).round(3)

        df_acao2.index = list(df_acao2['valor_real'])
        df_acao2.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)



        fig_4m, ax_4m = plt.subplots(figsize = (10,5))
        #plt.figure(figsize=(15,5))
        ax_4m = sns.heatmap(df_acao2, cbar_kws={'label': 'Frequência normalizada'})
        ax_4m.set_title('Previsão de '+(acao[:-3])+' para o mês (treino 4 meses)')
        ax_4m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        ax_4m.set_ylabel('Valor em R$');
        st.pyplot(fig_4m, use_container_width = True)


        #previsao 2m
        df_acao3 = pd.DataFrame(np.transpose(previsao_2m[dicionario[acao],:,:])).iloc[::-1]
        df_acao3['indice_bruto'] = list(df_acao3.index)
        df_acao3['valor_normalizado'] = amplitude*(df_acao3['indice_bruto'] - n_particoes)/(n_particoes)
        df_acao3['valor_real'] = (df_acao3['valor_normalizado']*dados_varias_2m[acao].std() + dados_varias_2m[acao].mean()).round(3)

        df_acao3.index = list(df_acao3['valor_real'])
        df_acao3.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)



        fig_2m, ax_2m = plt.subplots(figsize = (10,5))
        #plt.figure(figsize=(15,5))
        ax_2m = sns.heatmap(df_acao3, cbar_kws={'label': 'Frequência normalizada'})
        ax_2m.set_title('Previsão de '+(acao[:-3])+' para MEIO mês (treino 2 meses)')
        ax_2m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        ax_2m.set_ylabel('Valor em R$');
        st.pyplot(fig_2m, use_container_width = True)




    with col3:
        st.header('Comentários para '+ acao[:-3] +' no próximo mês')
        st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec auctor ex at feugiat commodo. Donec vitae lobortis mi. Integer lorem turpis, pulvinar non turpis at, lacinia eleifend ex. Etiam vel massa id sapien congue eleifend at eu libero. Nulla euismod gravida urna, sit amet lobortis sem. Morbi a odio orci. Donec dictum volutpat gravida. Sed nec consequat justo.\n Quisque dictum facilisis risus at scelerisque. Nullam tortor felis, feugiat ac posuere non, pharetra a odio. Sed purus risus, hendrerit sit amet fermentum sed, tempor a mauris. Integer pellentesque diam at nibh tincidunt, id fermentum lorem finibus. Vivamus vel erat vel mi ultrices maximus. Mauris risus nisi, porttitor sed pharetra nec, lacinia ac tortor. Nunc gravida facilisis quam at pretium. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Sed ac neque sit amet sapien elementum varius. Etiam libero lorem, laoreet eu venenatis nec, sollicitudin et lacus. Donec velit odio, dictum sed ligula ac, porta consequat velit. Curabitur suscipit lorem non mi dignissim, ac eleifend tortor dapibus. Aliquam erat volutpat.')
        st.write('Este é apenas um exemplo de Dashboard. Apesar dos resultados serem autênticos do meu modelo, não faço comentários extras aqui.')
        
        
with aba_margem:
    
    col1, col2, col3 = st.columns(3)


    with col1:
        #ultimo mes
        fig_ultimo_mes = px.line(dados_varias[acao] , range_x=[-40,0] ,width = 1176, height=450, title = f'Variação de {acao} no último mês (tempo zero é o presente)')
        fig_ultimo_mes.update_layout(xaxis_title = 'Tempo (dt = 2 equivale a um dia)')
        fig_ultimo_mes.update_layout(yaxis_title = 'Valor (R$)')
        st.plotly_chart(fig_ultimo_mes, use_container_width = True)

        #previsão 6m
        #df_acao = pd.DataFrame(np.transpose(previsao_6m[dicionario[acao],:,:])).iloc[::-1]
        #df_acao['indice_bruto'] = list(df_acao.index)
        #df_acao['valor_normalizado'] = amplitude*(df_acao['indice_bruto'] - n_particoes)/(n_particoes)
        #df_acao['valor_real'] = (df_acao['valor_normalizado']*dados_varias_6m[acao].std() + dados_varias_6m[acao].mean()).round(3)

        #df_acao.index = list(df_acao['valor_real'])
        #df_acao.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)
        
        valores1 = np.array(df_acao.index)
        probabilidades1 = np.transpose(np.array(df_acao))
        media1 = np.dot(probabilidades1,valores1)
        desvio_padrao1 = np.sqrt(np.dot(probabilidades1,valores1*valores1) - media1*media1)
        
        resultado1 = pd.DataFrame(media1, columns = ['Média'])
        resultado1['Limite Otimista'] = media1 + 2*desvio_padrao1 
        resultado1['Limite Pessimista'] = media1 - 2*desvio_padrao1 
        
        fig_media_desvio1 = px.line(resultado1, title = 'Previsão de '+(acao[:-3])+' para o mês (treino 6 meses)')
        fig_media_desvio1.update_layout(xaxis_title = 'Tempo (dt = 2 equivale a um dia)')
        fig_media_desvio1.update_layout(yaxis_title = 'Valor (R$)')
        st.plotly_chart(fig_media_desvio1, use_container_width = True)



        #fig_6m, ax_6m = plt.subplots( figsize = (15,8))
        #plt.figure(figsize=(15,5))
        #ax_6m = sns.heatmap(df_acao, cbar_kws={'label': 'Frequência normalizada'})
        #ax_6m.set_title('Previsão de '+(acao[:-3])+' para o mês (treino 6 meses)')
       # ax_6m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        #ax_6m.set_ylabel('Valor em R$');
        #st.pyplot(fig_6m, use_container_width = True)


    with col2:
        # previsao 4m
        #df_acao2 = pd.DataFrame(np.transpose(previsao_4m[dicionario[acao],:,:])).iloc[::-1]
        #df_acao2['indice_bruto'] = list(df_acao2.index)
        #df_acao2['valor_normalizado'] = amplitude*(df_acao2['indice_bruto'] - n_particoes)/(n_particoes)
        #df_acao2['valor_real'] = (df_acao2['valor_normalizado']*dados_varias_4m[acao].std() + dados_varias_4m[acao].mean()).round(3)

        #df_acao2.index = list(df_acao2['valor_real'])
        #df_acao2.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)

        valores2 = np.array(df_acao2.index)
        probabilidades2 = np.transpose(np.array(df_acao2))
        media2 = np.dot(probabilidades2,valores2)
        desvio_padrao2 = np.sqrt(np.dot(probabilidades2,valores2*valores2) - media2*media2)
        
        resultado2 = pd.DataFrame(media2, columns = ['Média'])
        resultado2['Limite Otimista'] = media2 + 2*desvio_padrao2 
        resultado2['Limite Pessimista'] = media2 - 2*desvio_padrao2 
        
        fig_media_desvio2 = px.line(resultado2, title = 'Previsão de '+(acao[:-3])+' para o mês (treino 4 meses)')
        fig_media_desvio2.update_layout(xaxis_title = 'Tempo (dt = 2 equivale a um dia)')
        fig_media_desvio2.update_layout(yaxis_title = 'Valor (R$)')
        st.plotly_chart(fig_media_desvio2, use_container_width = True)

        #fig_4m, ax_4m = plt.subplots(figsize = (15,8))
        #plt.figure(figsize=(15,5))
        #ax_4m = sns.heatmap(df_acao2, cbar_kws={'label': 'Frequência normalizada'})
        #ax_4m.set_title('Previsão de '+(acao[:-3])+' para o mês (treino 4 meses)')
        #ax_4m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        #ax_4m.set_ylabel('Valor em R$');
        #st.pyplot(fig_4m, use_container_width = True)


        #previsao 2m
        #df_acao3 = pd.DataFrame(np.transpose(previsao_2m[dicionario[acao],:,:])).iloc[::-1]
        #df_acao3['indice_bruto'] = list(df_acao3.index)
        #df_acao3['valor_normalizado'] = amplitude*(df_acao3['indice_bruto'] - n_particoes)/(n_particoes)
        #df_acao3['valor_real'] = (df_acao3['valor_normalizado']*dados_varias_2m[acao].std() + dados_varias_2m[acao].mean()).round(3)

        #df_acao3.index = list(df_acao3['valor_real'])
        #df_acao3.drop(['indice_bruto','valor_normalizado','valor_real'],axis=1,inplace = True)

        valores3 = np.array(df_acao3.index)
        probabilidades3 = np.transpose(np.array(df_acao3))
        media3 = np.dot(probabilidades3,valores3)
        desvio_padrao3 = np.sqrt(np.dot(probabilidades3,valores3*valores3) - media3*media3)
        
        resultado3 = pd.DataFrame(media3, columns = ['Média'])
        resultado3['Limite Otimista'] = media3 + 2*desvio_padrao3 
        resultado3['Limite Pessimista'] = media3 - 2*desvio_padrao3 
        
        fig_media_desvio3 = px.line(resultado3, title = 'Previsão de '+(acao[:-3])+' para MEIO mês (treino 2 meses)')
        fig_media_desvio3.update_layout(xaxis_title = 'Tempo (dt = 2 equivale a um dia)')
        fig_media_desvio3.update_layout(yaxis_title = 'Valor (R$)')
        st.plotly_chart(fig_media_desvio3, use_container_width = True)

        #fig_2m, ax_2m = plt.subplots(figsize = (15,8))
        #plt.figure(figsize=(15,5))
        #ax_2m = sns.heatmap(df_acao3, cbar_kws={'label': 'Frequência normalizada'})
        #ax_2m.set_title('Previsão de '+(acao[:-3])+' para MEIO mês (treino 2 meses)')
        #ax_2m.set_xlabel('Tempo ($\Delta t = 2$ equivale a 1 dia)')
        #ax_2m.set_ylabel('Valor em R$');
        #st.pyplot(fig_2m, use_container_width = True)




    with col3:
        st.header('Comentários para '+ acao[:-3] +' no próximo mês')
        st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec auctor ex at feugiat commodo. Donec vitae lobortis mi. Integer lorem turpis, pulvinar non turpis at, lacinia eleifend ex. Etiam vel massa id sapien congue eleifend at eu libero. Nulla euismod gravida urna, sit amet lobortis sem. Morbi a odio orci. Donec dictum volutpat gravida. Sed nec consequat justo.\n Quisque dictum facilisis risus at scelerisque. Nullam tortor felis, feugiat ac posuere non, pharetra a odio. Sed purus risus, hendrerit sit amet fermentum sed, tempor a mauris. Integer pellentesque diam at nibh tincidunt, id fermentum lorem finibus. Vivamus vel erat vel mi ultrices maximus. Mauris risus nisi, porttitor sed pharetra nec, lacinia ac tortor. Nunc gravida facilisis quam at pretium. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Sed ac neque sit amet sapien elementum varius. Etiam libero lorem, laoreet eu venenatis nec, sollicitudin et lacus. Donec velit odio, dictum sed ligula ac, porta consequat velit. Curabitur suscipit lorem non mi dignissim, ac eleifend tortor dapibus. Aliquam erat volutpat.')
        st.write('Este é apenas um exemplo de Dashboard. Apesar dos resultados serem autênticos do meu modelo, não faço comentários extras aqui.')
    
with aba_sobreoautor:
    
    col1,col2,col3 = st.columns(3)
    
    with col2:
        st.header('Sobre o autor Mário Henrique Figlioli Donato')
        st.write('Mário H. F. Donato é Bacharel e Mestre em Física pelo Instituto de Física de São Carlos da USP - Campus São Carlos. Atualmente é Doutorando em Física pela mesma instituição, em estágio final da conclusão da sua tese. Possui quase 10 anos de experiência em tópicos avançados de Matemática e Física, em particular relacionados à modelagem de sistemas estocásticos (isto é, "aleatórios"). Após 5 anos de experiência no estudo de sistemas clássicos e quânticos abertos, está estudando maneiras de aplicar técnicas de Machine Learning (em particular, Deep Learning) para a previsibilidade de séries temporais. Aqui citamos: evolução do valor de uma carteira de investimentos, previsibilidade da demanda de produção em indústrias, produtividade de plantações e criações de animais (método em desenvolvimento).')
        
        st.header('Contato')
        st.write('mariohfdonato@gmail.com')
        st.link_button("LinkedIn", "https://www.linkedin.com/in/mario-henrique-figlioli-donato-86a42b208/")