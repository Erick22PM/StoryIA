# utils/fijar_cpu.py
import os

def forzar_cpu():
    # Deshabilitar TODA la GPU para TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Evitar XLA (que está forzando a compilar kernels)
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

    # Evitar que TensorFlow use aceleradores tipo OneDNN que disparan kernels en GPU
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
