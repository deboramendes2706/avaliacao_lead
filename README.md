# Avaliacao_lead
 ### Descrição do projeto
 O projeto desenvolvido nessa avaliação consta na leitura de comentários do twitter, que estão armazenado em um arquivo do tipo csv, e na predição da toxicidade deles, tal forma que, podem ser tóxicos ou não tóxicos.
 
## Organização do repositório
 ├──  [data](data) -> Pasta contendo os arquivos salvos durante a execução dos códigos desenvolvidos no projeto e arquivos necessários para essa execução.
   - [arquivos_resultados](/data/arquivos_resultados) -> Pasta contendo todos os arquivos csv salvos durante a execução do código
     - [Arquivo csv com os resultados da análise de todas as colunas, criadas à partir dos pré-processamentos, utilizando o melhor modelo](data/arquivos_resultados/df_total_results.csv)
     - Todos os outros arquivos são os resultados da análise de cada coluna pré-processada
   - [classificadores](/data/classificadores) -> Arquivo onde serão realizados todos os pré-processamentos e testes de modelos.
     - [Arquivo contendo o melhor classificador.](data/classificadores/Bayes_BOW_stemm_comments_best_model.pickle)
   - [Arquivo csv contendo os dados de trieno](data/train_binary_small.csv)
   - [Arquivo csv contendo os dados de teste](data/test_binary_small.csv)


 ├──  [body.ipnyb](body.ipynb) -> Arquivo onde serão realizados todos os pré-processamentos e testes de modelos.
 
 ├──  [config.py](config.py) -> Arquivo no qual iremos configurar os hiperparâmetros dos nossos modelos e seus inicializadores.
 
 ├──  [config_links.py](config_links.py) -> Arquivo utlizado para configurar todos os caminhos dos arquivos utilizados durante o projeto.
 
 ├──  [functions.py](functions.py) -> Arquivo onde armazenamos todas as funções utilizadas no projeto.
 
 ├──  [script_de_classificação.py](script_de_classificação.py) -> Arquivo contendo o passo a passo para a leitura de um arquivo csv,contendo dados de teste, utilizando o melhor modelo e o melhor processamento definidos.
 
 ├──  [requirements.txt](requirements.txt) -> Arquivo contendo as bibliotecas necessárias para a utilização dos arquivo do repositório
 
