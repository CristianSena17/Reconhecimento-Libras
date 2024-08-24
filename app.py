import csv
import copy
from collections import Counter
from collections import deque
import itertools
from typing import List, Mapping, Optional, Tuple, Union
import pandas as pd
import cv2 as cv
import numpy as np
import mediapipe as mp
import os

from model import ClassificarGestos
import draw


def main():
    # Parâmetros de configuração
    camera = 1
    largura_camera = 960
    altura_camera = 540

    usar_borda = True
    video_path = 'data/videos/Video01.mp4'
    # Configuração da câmera
    captura = cv.VideoCapture(camera)
    captura.set(cv.CAP_PROP_FRAME_WIDTH, largura_camera)
    captura.set(cv.CAP_PROP_FRAME_HEIGHT, altura_camera)

    # Carregamento do modelo
    mp_hand = mp.solutions.hands
    hand = mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    classificador_gestos = ClassificarGestos()

    # Leitura dos rótulos
    with open('model/gesture_sign_classifier/gesture_sign_classifier_label.csv', encoding='utf-8-sig') as f:
        rotulos_gestos = [row[0] for row in csv.reader(f) if row]
   

    # Histórico de coordenadas
    comprimento_historico = 46
    hand_marks = 21
    historico_pontos = [deque(maxlen=comprimento_historico) for _ in range(hand_marks)]

    # Histórico de gestos dos dedos
    historico_gestos_dedos = deque(maxlen=46)

    modo = 0

    while True:
        

        # Processamento da tecla (ESC: finalizar)
        tecla = cv.waitKey(10)
        if tecla == 27:  # ESC
            break
        numero, modo = selecionar_modo(tecla, modo)

        # Captura da câmera
        ret, imagem = captura.read()
        if not ret:
            break
        # imagem = cv.flip(imagem, 1)  # Exibir espelhado
        imagem_debug = copy.deepcopy(imagem)

        # Implementação da detecção
        imagem = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)
        imagem.flags.writeable = False
        resultados = hand.process(imagem)
        imagem.flags.writeable = True

        # Processamento dos resultados da detecção
        if resultados.multi_hand_landmarks:
            for pontos_mao, mao_dominante in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
                # Cálculo dos pontos de referência
                lista_pontos = calcular_lista_pontos(imagem_debug, pontos_mao)
                retangulo = calcula_retangulo_delimitador(lista_pontos)

                # Processamento do histórico de pontos
                historico_processado = processamento_combinado(imagem_debug, historico_pontos, modo)
                gravar_csv(numero, modo, historico_processado)

                for i in range(len(historico_pontos)):
                    historico_pontos[i].append(lista_pontos[i])

                # Classificação do gesto dos dedos
                id_gesto_dedo = 0
                if len(historico_processado) == ((comprimento_historico * hand_marks) * 4):
                    id_gesto_dedo = classificador_gestos(historico_processado)

                # Histórico de gestos
                historico_gestos_dedos.append(id_gesto_dedo)
                gesto_comum = Counter(historico_gestos_dedos).most_common()

                # Desenho na imagem
                imagem_debug = draw_landmarks_on_image(
                    imagem_debug,
                    pontos_mao,
                    draw.HAND_CONNECTIONS,
                    draw.get_default_hand_landmarks_style(),
                    draw.get_default_hand_connections_style()
                )
                imagem_debug = desenhar_retangulo_borda(
                    usar_borda, imagem_debug, retangulo, mao_dominante,
                    rotulos_gestos[gesto_comum[0][0]],  modo, numero)

        else:
            for i in range(len(historico_pontos)):
                historico_pontos[i].append([0, 0])

        # Exibição da tela
        cv.imshow('Reconhecimento de Gestos da Mao', imagem_debug)

    captura.release()
    cv.destroyAllWindows()

def selecionar_modo(tecla, modo):
    numero = -1

    # Converte teclas de '0' a '9', de 'a' a 'z' e 'ç' em 6
    if 48 <= tecla <= 57:  # 0 ~ 9
        numero = tecla - 48
    elif 97 <= tecla <= 122:  # a ~ z
        numero = tecla - 87
    elif tecla == 231:  # ç
        numero = 6


    # Utiliza match-case para alternar modos
    match tecla:
        case 9:  # TAB
            modo = 0
        case 32:  # ESPAÇO
            modo = 2
        case 44:  # Vírgula
            modo = 1
    
    return numero, modo



# Este código foi adaptado a partir do repositório MediaPipe.
# Alterações realizadas:
# 1. Renomeação da Função e Variaveis 
# 2. Simplificação da Lógica: A função foi simplificada removendo a validação de valores normalizados
# 3. Iteração Direta: A iteração sobre os pontos foi feita diretamente
# 4. Remoção de Validação de Coordenadas: A validação de valores normalizados foi removida para simplificação.
# Fonte: MediaPipe Authors. (2020). MediaPipe Python (Version 2.0) [Source code]. GitHub. https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python
def calcular_lista_pontos(imagem, pontos_landmarks):
    largura_imagem, altura_imagem = imagem.shape[1], imagem.shape[0]
    lista_pontos = []

    for ponto in pontos_landmarks.landmark:
        # Converte coordenadas normalizadas em coordenadas de pixel
        x = min(int(ponto.x * largura_imagem), largura_imagem - 1)
        y = min(int(ponto.y * altura_imagem), altura_imagem - 1)
        lista_pontos.append([x, y])


    return lista_pontos




def pre_processar_pontos(historico_pontos):
    # Cria uma cópia dos pontos históricos
    pontos_temp = [list(deque) for deque in historico_pontos]

    # Inicializa as listas para armazenar as colunas concatenadas
    colunas_concatenadas = [[] for _ in range(len(pontos_temp[0]))]

    # Itera sobre cada deque e extrai as colunas
    for deque in pontos_temp:
        for i, par in enumerate(deque):
            colunas_concatenadas[i].append(par)

    # Função para normalizar um array
    def normalizar(array):
        max_valor = max(max(map(abs, par)) for par in array)
        if max_valor == 0:
            return array
        return [[coord / max_valor for coord in par] for par in array]

    # Processa cada coluna extraída
    colunas_processadas = []
    for coluna in colunas_concatenadas:
        base_x, base_y = coluna[0][0], coluna[0][1]
        coluna_relativa = [[x - base_x, y - base_y] for x, y in coluna]
        coluna_normalizada = normalizar(coluna_relativa)
        colunas_processadas.append(coluna_normalizada)

    # Converte as colunas processadas em um array unidimensional
    array_unidimensional = [coord for coluna in colunas_processadas for par in coluna for coord in par]

    return array_unidimensional


def processar_historico_pontos(imagem, historico_pontos, modo):
    largura_imagem, altura_imagem = imagem.shape[1], imagem.shape[0]

    historico_pontos_processados = []

    for lista_pontos in historico_pontos:
        nova_lista_pontos = copy.deepcopy(lista_pontos)

        base_x, base_y = 0, 0
        for i, ponto in enumerate(nova_lista_pontos):
            if modo != 2:
                if i == 0:
                    base_x, base_y = ponto[0], ponto[1]

                # Normaliza as coordenadas em relação ao ponto base
                nova_lista_pontos[i][0] = (nova_lista_pontos[i][0] - base_x) / largura_imagem
                nova_lista_pontos[i][1] = (nova_lista_pontos[i][1] - base_y) / altura_imagem
            else:
                nova_lista_pontos[i][0] = 0
                nova_lista_pontos[i][1] = 0
        
        # Achata a lista de pontos
        nova_lista_pontos = list(itertools.chain.from_iterable(nova_lista_pontos))
        historico_pontos_processados.append(nova_lista_pontos)

    # Concatenando todos os valores em um único array
    array_concatenado = np.concatenate(historico_pontos_processados)   
   
    return array_concatenado



def processamento_combinado(image, point_history, mode , debug=False):
    # Processa as landmarks
    landmark_process = pre_processar_pontos(point_history)

    # Processa o histórico dos pontos
    point_history_process = processar_historico_pontos(image, point_history, mode)

    #Alterna os valores entre os resultados das duas funções
    combined_result = []
    min_length = min(len(landmark_process), len(point_history_process)) // 2
    
    j = len(point_history)
    cont = 1
    conti=0
    h = 0
    c = 0

    
    for i in range(min_length):
        if cont == j:
            c += 1
            h = c
            cont = 1

        if cont == 1:
            combined_result.append(landmark_process[i])
  
            combined_result.append(landmark_process[i+1])

            if h < len(point_history_process):
                combined_result.append(point_history_process[h])

            if h + 1 < len(point_history_process):
                combined_result.append(point_history_process[h+1])

            c+=1
        else:
            h += j*2
            combined_result.append(landmark_process[i+conti])

            combined_result.append(landmark_process[i+conti+1])

            if h < len(point_history_process):
                combined_result.append(point_history_process[h])

            if h + 1 < len(point_history_process):
                combined_result.append(point_history_process[h+1])

        
        cont += 1
        conti +=1

    
    return np.array(combined_result)


def gravar_csv(numero, modo, lista_gestos):
    # Verifica se o modo é 1 ou 2 e se o número está no intervalo válido
    if (modo in [1, 2]) and (0 <= numero <= 35):
        # Define o caminho do arquivo CSV
        caminho_csv = 'model/gesture_sign_classifier/gesture_sign_history.csv'
        # Abre o arquivo CSV no modo de adição
        with open(caminho_csv, 'a', newline="") as arquivo:
            escrita = csv.writer(arquivo)
            # Escreve a linha de dados no arquivo CSV
            escrita.writerow([numero , *lista_gestos])
    return


# Este código foi retirado a partir do repositório MediaPipe.
# Fonte: MediaPipe Authors. (2020). MediaPipe Python (Version 2.0) [Source code]. GitHub. https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/python
def draw_landmarks_on_image(
    image: np.ndarray,
    landmark_list: any,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[draw.DrawingSpec, Mapping[int, draw.DrawingSpec]] = draw.DrawingSpec(color=(255, 0, 0) ),
    connection_drawing_spec: Union[draw.DrawingSpec, Mapping[Tuple[int, int], draw.DrawingSpec]] = draw.DrawingSpec(),
    is_drawing_landmarks: bool = True
) -> np.ndarray:
    """Draws the landmarks and the connections on the image."""
    
    WHITE_COLOR = (255, 255, 255)
    _BGR_CHANNELS = 3
    _VISIBILITY_THRESHOLD = 0.5
    _PRESENCE_THRESHOLD = 0.5
    if not landmark_list:
        return image

    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel BGR data.')

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = draw._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx],
                        drawing_spec.color, drawing_spec.thickness)

    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            circle_border_radius = max(
                drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
            cv.circle(image, landmark_px, circle_border_radius,
                      WHITE_COLOR, drawing_spec.thickness)
            cv.circle(image, landmark_px, drawing_spec.circle_radius,
                      drawing_spec.color, drawing_spec.thickness)

    return image

# Este código foi adaptado a partir de uma resposta encontrada no Stack Overflow.
# Fonte original: Stack Overflow, "Create a rectangle around all the points returned from mediapipe hand landmark detection", por mpk732, disponível em:
# https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
def calcula_retangulo_delimitador(pontos_landmarks):
    # Calcula a caixa delimitadora
    x_min = min(p[0] for p in pontos_landmarks)
    y_min = min(p[1] for p in pontos_landmarks)
    x_max = max(p[0] for p in pontos_landmarks)
    y_max = max(p[1] for p in pontos_landmarks)

    return [x_min, y_min, x_max, y_max]


def desenhar_retangulo_borda(usando_borda, imagem, borda, mao_dominante, texto_gesto_mao, modo, numero):
    if usando_borda:
        # Desenha um retângulo ao redor da mão
        cv.rectangle(imagem, (borda[0], borda[1]), (borda[2], borda[3]), (0, 0, 0), 1)
        # Desenha um retângulo para exibir informações
        cv.rectangle(imagem, (borda[0], borda[1]), (borda[2], borda[1] - 22), (0, 0, 0), -1)
        # Exibe a mão dominante (esquerda ou direita)
        texto_info = mao_dominante.classification[0].label
        cv.putText(imagem, texto_info, (borda[0] + 5, borda[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Exibe o texto do gesto da mão se houver
    if texto_gesto_mao:
        cv.putText(imagem, "Gesto da Mao:" + texto_gesto_mao, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(imagem, "Gesto da Mao:" + texto_gesto_mao, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)    


    # Exibe o modo atual
    modos = ['Gestos em Movimento', 'Gestos Estaticos']
    if modo in [1, 2]:
        cv.putText(imagem, "MODO:" + modos[modo - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        # Exibe a tecla pressionada (0-9 ou a-z)
        if 0 <= numero <= 9:  # De 0 a 9
            cv.putText(imagem, "TECLA PRESSIONADA:" + str(numero), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        elif 10 <= numero <= 35:  # De 'a' a 'z'
            cv.putText(imagem, "TECLA PRESSIONADA:" + chr(numero + 87), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return imagem



if __name__ == '__main__':
    main()
