# Reconhecimento da Linguagem LIBRAS e Outros Gestos da Mão

## Descrição do Projeto

Este projeto utiliza o MediaPipe para detectar e classificar gestos das mãos em tempo real utilizando uma câmera. Foi desenvolvido para capturar gestos e gravá-los em arquivos CSV para análise posterior. O sistema emprega um modelo pré-treinado para classificar os gestos das mãos e possui funcionalidades de detecção de mãos, processamento de gestos e exibição de resultados. Assim, criamos uma base de dados que reconhece a linguagem LIBRAS e outros 5 gestos adicionais.

### Gestos Reconhecidos
- Alfabeto completo em LIBRAS
- Outros gestos:
  - Relógio (Sentido Horário)
  - Tchau (Mexer a mão aberta 2 vezes para cada lado, começando pela direita)
  - Não (Mexer a mão e o dedo 2 vezes para cada lado, começando pela direita)
  - Oi (LIBRAS)
  - Tudo Bem? (LIBRAS)

## Funcionalidades

- Detecção e rastreamento de mãos em tempo real.
- Classificação de gestos utilizando um modelo pré-treinado.
- Registro de gestos em arquivos CSV.
- Exibição das coordenadas dos pontos de referência das mãos e das conexões entre eles.
- Alternância entre diferentes modos de operação utilizando teclas específicas.
- Geração de dados a partir de imagens.
- Geração de dados pela prórpia webcam.
- Criação de modelos de gestos treinados.

## Requisitos

- Python 3.7+
- OpenCV
- Mediapipe
- NumPy
- Pandas

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/GuilhermeMPG/reconhecimento-linguagens-sinais-gestos.git
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd seu-repositorio
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
## Como Executar

1. Conecte uma câmera ao seu computador.
2. Execute o script principal:
   ```bash
   python app.py
   ```
3. Utilize as teclas especificadas para alternar entre os modos e registrar gestos.
4. A imagem capturada será exibida em uma janela com os pontos de referência e conexões desenhados, bem como informações sobre o gesto classificado e o modo atual.

### Seleção de Modos

A função `selecionar_modo(tecla, modo)` permite alternar entre diferentes modos de operação utilizando as seguintes teclas:

- **TAB (Tecla 9)**: Alterna para o modo 0 (Modo padrão).
- **VÍRGULA (Tecla 32)**: Alterna para o modo 1 (Registro de histórico de gestos).
- **ESPAÇO (Tecla 44)**: Alterna para o modo 2 (Histórico de gestos estáticos).

Além disso, as teclas de '0' a '9' e de 'a' a 'z' podem ser usadas para registrar gestos específicos:

- **'0' a '9'**: As teclas numéricas de '0' a '9' são convertidas para os números correspondentes.
- **'a' a 'z'**: As teclas alfabéticas de 'a' a 'z' são convertidas para os números de 10 a 35.

## Estrutura do Projeto

- `app.py`: Script principal responsável por executar o sistema de reconhecimento de gestos em tempo real e a gravação de dados.
- `model/`: Diretório contendo o modelo e arquivos relacionados à classificação de gestos.
  - `model/gesture_sign_classifier.py`: Código que executa o reconhecimento em tempo real utilizando o modelo de classificação de gestos.
- `draw.py`: Script para desenhar os pontos de referência e conexões nas imagens capturadas.
- `creatData.py`: Código responsável por gerar os dados no arquivo .csv a partir das imagens capturadas.

## Data_Base
- Arquivo .csv gerado: [gesture_sign_history.csv](https://drive.google.com/file/d/144Ct4juy9VBNN4yT2DzGJG20vMGR42ug/view?usp=sharing) 
- Imagens Externas: https://www.kaggle.com/datasets/williansoliveira/libras
## Método Adotado 
### Estrutura dos Dados Gerados
Vamos destacar como é desenvolvido os dados que serão o utilizados para gerar nosso modelo. Para isso vamos trabalhar basiciamente com esses dois conjuntos de valores:

- Pontos da Mão: 21
- Histórico: 46

Para identificar gestos estático, utilizamos uma técnica lógica: selecionamos um ponto base na mão e calculamos a distância entre esse ponto e os outros. Dessa forma, obtemos um padrão de valores que identifica o gesto. Realizamos esse cálculo para os 21 pontos da mão.

Então, surge a pergunta: para que serve o histórico, se apenas com esse cálculo podemos obter um padrão? A resposta é simples: isso não funciona para gestos que exigem movimento. 

Para gestos em movimento, calculamos essa variação ao longo do tempo. Assim, criamos um histórico de movimento, foi definindo um histórico de 46 que nos dá tempo suficiente para registrar gestos curtos, mas efetivos. Esse valor pode ser ajustado conforme necessário.

### Estrutura do Array

Para unir essas ideias, criamos um array de 966 (21 vezes 46) pontos , onde cada ponto da mão possui dois valores(X e Y) para a posição e outros dois para o movimento no espaço. Assim, a base é treinada com arrays de tamanho 966 vezes 4. Quando o gesto é estático, os valores de movimento no espaço é igual a zero.

- Exemplo de Estrutura

Considere:
- **Pontos da Mão igual a 3**: `[[px1, px2, px3], [py1, py2, py3], [pz1, pz2, pz3], [pv1, pv2, pv3], [ps1, ps2, ps3]]`
- **Histórico igual a 5**: `[[x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5], [z1, z2, z3, z4, z5]]`

Concatenamos os dois arrays de forma que a posição represente o ponto da mão corretamente:

- **Array Final**: 
```
[
    [px1, x1], [px2, y1], [px3, z1],
    [py1, x2], [py2, y2], [py3, z2],
    [pz1, x3], [pz2, y3], [pz3, z3],
    [pv1, x4], [pv2, y4], [pv3, z4],
    [ps1, x5], [ps2, y5], [ps3, z5]
]
```


Assim, obtemos um array com 30 valores, lembrando que cada valor representa X e Y. Portanto, temos 30 vezes 4, resultando em um total de 120 valores.

Dado que temos 966 pontos no total, a matriz final utilizada será de 966 vezes 4, resultando em 3864 valores por linha de dados.

---


  

## Funções Principais Do Main

### main()

A função principal que configura a câmera, carrega o modelo, lê os rótulos dos gestos e inicia o loop de captura de vídeo para detecção e classificação de gestos.

### selecionar_modo(tecla, modo)

Permite alternar entre diferentes modos de operação e registrar gestos específicos com base nas teclas pressionadas.

### calcular_lista_pontos(imagem, pontos_landmarks)

Converte coordenadas normalizadas em coordenadas de pixel para a imagem.

### pre_processar_pontos(historico_pontos)

Pré-processa os pontos históricos para normalização e preparação para a classificação.

### processar_historico_pontos(imagem, historico_pontos, modo)

Processa o histórico de pontos para normalização em relação ao ponto base.

### processamento_combinado(imagem, historico_pontos, modo)

Combina os resultados das funções de pré-processamento e processamento do histórico de pontos.

### gravar_csv(numero, modo, lista_gestos)

Grava os gestos registrados em um arquivo CSV.

### draw_landmarks_on_image(image, landmark_list, connections, ...)

Desenha os pontos de referência e as conexões nas imagens capturadas.

### calcula_retangulo_delimitador(pontos_landmarks)

Calcula a caixa delimitadora para os pontos de referência das mãos.

### desenhar_retangulo_borda(usando_borda, imagem, borda, mao_dominante, texto_gesto_mao, modo, numero)

Desenha um retângulo ao redor da mão e exibe informações sobre a mão dominante e o gesto classificado.

## Funções Adicionais 

### Classificação
A classe `ClassificarGestos` utiliza o TensorFlow Lite para classificar gestos das mãos. Inicialmente, carrega um modelo TFLite especificado pelo caminho, define um limiar de confiança e um valor para resultados inválidos. No método `__call__`, a classe recebe pontos históricos da mão, define o tensor de entrada do modelo, processa a entrada com o interpretador, obtém o resultado da classificação e verifica se a confiança é suficiente. Se a confiança for inferior ao limiar, retorna um valor inválido; caso contrário, retorna o índice do gesto classificado.

### CreatData
O script principal utiliza MediaPipe para detectar e processar gestos das mãos em imagens. Inicialmente, configura parâmetros de confiança para detecção e rastreamento. Em seguida, inicializa o MediaPipe Hands para detectar uma mão por vez e define um histórico de coordenadas utilizando deque.

O código percorre um diretório de imagens, selecionando até 100 arquivos aleatórios de cada subpasta. Para cada imagem, realiza a detecção dos landmarks das mãos, atualiza o histórico de coordenadas e processa esses dados. Os resultados processados são então gravados em arquivos CSV para análise posterior. O mapeamento de letras para números é feito através de um dicionário específico.

A função main() coordena todo o processo, desde a configuração inicial até a gravação dos resultados, sendo executada quando o script é rodado.

### Arquivo Jupyter

Este notebook Jupyter serve como um guia passo a passo para a criação de um modelo de reconhecimento de gestos utilizando dados previamente coletados. Ele demonstra o processo completo, desde a preparação dos dados até o treinamento do modelo com Keras e a conversão para o formato TFLite.


## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e correções.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Referências

- MediaPipe Authors. (2020). MediaPipe Python (Version 2.0) [Source code]. GitHub. [Link](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python)
- Stack Overflow. (2021). Create a rectangle around all the points returned from MediaPipe hand landmark detection. [Link](https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d)
- MediaPipe Authors. (2020). Recognition using MediaPipe (Version 2.0) [Source code]. GitHub.
- OpenCV Documentation. (2021). OpenCV-Python Tutorials. [Link](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- Google AI. MediaPipe Solutions Guide. [Link](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=pt-br)
- TensorFlow. (2021). TensorFlow Lite: Guia de inferência. [Link](https://www.tensorflow.org/lite/guide/inference?hl=pt-br#load_and_run_a_model_in_python)
- TensorFlow. (2021). TensorFlow Bibliography. [Link](https://www.tensorflow.org/about/bib?hl=pt-br)
- IPnet Growth Partner. (2021). Padronização e Normalização de Dados em Machine Learning. Medium. [Link](https://medium.com/ipnet-growth-partner/padronizacao-normalizacao-dados-machine-learning-f8f29246c12)

---

Este README fornece uma visão geral do projeto, suas funcionalidades, instruções de instalação e execução, bem como uma explicação detalhada sobre os modos de operação e as funções principais do código.
