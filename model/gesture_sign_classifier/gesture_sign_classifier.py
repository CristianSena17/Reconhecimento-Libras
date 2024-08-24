import numpy as np
import tensorflow as tf

class ClassificarGestos:
    def __call__(self, historico_pontos):
        caminho_modelo='model/gesture_sign_classifier/gesture_sign_classifier.tflite'
        # Inicialização do modelo TFLite
        # Código alterado a partir de: https://www.tensorflow.org/lite/guide/inference?hl=pt-br#load_and_run_a_model_in_python
        self.interpreter = tf.lite.Interpreter(model_path=caminho_modelo, num_threads=1)
        self.interpreter.allocate_tensors()
        self.detalhes_entrada = self.interpreter.get_input_details()
        self.detalhes_saida = self.interpreter.get_output_details()
        self.limiar_confianca = 0.8
        self.valor_invalido = 0
        # Definir tensor de entrada
        # Código alterado a partir de: https://stackoverflow.com/questions/50443411/how-to-load-a-tflite-model-in-script
        indice_tensor_entrada = self.detalhes_entrada[0]['index']
        self.interpreter.set_tensor(indice_tensor_entrada, np.array([historico_pontos], dtype=np.float32))
        self.interpreter.invoke()

        # Obter resultado do modelo
        indice_tensor_saida = self.detalhes_saida[0]['index']
        resultado = self.interpreter.get_tensor(indice_tensor_saida)
        indice_resultado = np.argmax(np.squeeze(resultado))

        # Verificar a confiança do resultado
        if np.squeeze(resultado)[indice_resultado] < self.limiar_confianca:
            indice_resultado = self.valor_invalido

        return indice_resultado
