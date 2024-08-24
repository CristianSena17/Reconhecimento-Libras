import csv
import copy

from collections import Counter
from collections import deque
import os
import random

import cv2 as cv

import mediapipe as mp

from app import calcular_lista_pontos, processamento_combinado, gravar_csv


def main():
    # Configurações de confiança mínima para detecção e rastreamento
    min_detection_conf = 0.7
    min_tracking_conf = 0.5

    # Inicialização do MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_conf,
        min_tracking_confidence=min_tracking_conf
    )

    # Histórico de coordenadas
    comprimento_hist = 46
    comprimento_ponto = 21
    historico_pontos = [deque(maxlen=comprimento_hist)
                        for _ in range(comprimento_ponto)]

    # Diretório de imagens para processamento
    dir_imagens = "data/test"
   # Iterar pelas pastas dentro do diretório
    for pasta in os.listdir(dir_imagens):
        caminho_pasta = os.path.join(dir_imagens, pasta)

        # Obter a lista de arquivos na pasta e selecionar 100 aleatoriamente
        arquivos = os.listdir(caminho_pasta)
        # Seleciona 100 ou menos se houver menos de 100 arquivos
        arquivos_selecionados = random.sample(
            arquivos, min(100, len(arquivos)))

        for arquivo in arquivos:
            caminho_arquivo = os.path.join(caminho_pasta, arquivo)

            # Leitura e processamento da imagem
            imagem = cv.imread(caminho_arquivo)
            imagem_debug = copy.deepcopy(imagem)
            imagem = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)
            imagem.flags.writeable = False
            resultados = hands.process(imagem)
            imagem.flags.writeable = True

            if resultados.multi_hand_landmarks:
                for landmarks_mao, mao in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
                    lista_landmarks = calcular_lista_pontos(
                        imagem_debug, landmarks_mao)
                    for i in range(comprimento_hist):
                        for j in range(len(historico_pontos)):
                            historico_pontos[j].append(lista_landmarks[j])

                    lista_historico_processada = processamento_combinado(
                        imagem_debug, historico_pontos, 2)
                    gravar_csv(obter_numero_da_letra(pasta),
                                2, lista_historico_processada)

        print(pasta, obter_numero_da_letra(pasta))


def obter_numero_da_letra(letra):
    dicionario_letras = {
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
        'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25,
        'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
        'Y': 34, 'Z': 35
    }
    return dicionario_letras.get(letra, None)


if __name__ == '__main__':
    main()
