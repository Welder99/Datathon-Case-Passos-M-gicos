import logging
import joblib
import os
import pandas as pd

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configura e retorna um logger padronizado para o projeto.
    Garante que todos os módulos usem o mesmo formato de log.
    """
    logger = logging.getLogger(name)
    
    # Evita adicionar múltiplos handlers se a função for chamada várias vezes
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Handler para o console (exibe no terminal)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        
        # Formato do log: [Data/Hora] - [Nível] - [Nome_do_Módulo] - Mensagem
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
        c_handler.setFormatter(formatter)
        
        logger.addHandler(c_handler)
        
    return logger

def load_model(filepath: str):
    """Carrega um modelo serializado (.joblib)."""
    logger = setup_logger(__name__)
    if not os.path.exists(filepath):
        logger.error(f"Arquivo de modelo não encontrado: {filepath}")
        raise FileNotFoundError(f"Arquivo {filepath} não existe.")
    
    logger.info(f"Carregando modelo de: {filepath}")
    return joblib.load(filepath)

def save_model(model, filepath: str):
    """Salva um modelo treinado."""
    logger = setup_logger(__name__)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Modelo salvo com sucesso em: {filepath}")