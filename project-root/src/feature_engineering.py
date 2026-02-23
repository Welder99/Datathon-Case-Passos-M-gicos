import pandas as pd
import numpy as np
import logging

# Configuração de log
logger = logging.getLogger(__name__)

def create_academic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas variáveis (features) baseadas nas notas e indicadores da Passos Mágicos.
    
    Args:
        df (pd.DataFrame): DataFrame após a limpeza inicial do preprocessing.py.
        
    Returns:
        pd.DataFrame: DataFrame com as novas colunas adicionadas.
    """
    logger.info("Iniciando a engenharia de features (Feature Engineering)...")
    
    # Criamos uma cópia para evitar o aviso de SettingWithCopyWarning do Pandas
    df_feat = df.copy()
    
    # 1. Média das Notas Escolares
    # Calculamos a média ignorando os NaNs (skipna=True) para não penalizar 
    # injustamente o aluno caso falte apenas uma nota no registro.
    notas_cols = ['Mat', 'Por', 'Ing']
    df_feat['Media_Notas'] = df_feat[notas_cols].mean(axis=1, skipna=True)
    
    # 2. Média dos Indicadores Passos Mágicos
    # Esses indicadores são fundamentais para entender o aluno de forma holística.
    indicadores_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']
    df_feat['Media_Indicadores'] = df_feat[indicadores_cols].mean(axis=1, skipna=True)
    
    # 3. Flags de Risco Comportamental (Features Binárias)
    # Criamos alertas (1 ou 0) se indicadores críticos estiverem abaixo de 5.0.
    # IEG = Indicador de Engajamento | IAN = Indicador de Adequação ao Nivelamento
    # O uso do pd.to_numeric garante que não teremos erros de tipo ao fazer a comparação.
    df_feat['Alerta_Engajamento_Baixo'] = (pd.to_numeric(df_feat['IEG'], errors='coerce') < 5.0).astype(int)
    df_feat['Alerta_Adequacao_Baixa'] = (pd.to_numeric(df_feat['IAN'], errors='coerce') < 5.0).astype(int)
    
    # 4. Discrepância entre Notas e Indicador de Aprendizagem (IDA)
    # Alunos que vão bem na escola pública, mas mal na Passos Mágicos (ou vice-versa)
    # Essa feature pode ser um forte sinalizador de risco de evasão/defasagem.
    df_feat['Discrepancia_Notas_IDA'] = abs(df_feat['Media_Notas'] - pd.to_numeric(df_feat['IDA'], errors='coerce'))
    
    novas_colunas = [
        'Media_Notas', 'Media_Indicadores', 
        'Alerta_Engajamento_Baixo', 'Alerta_Adequacao_Baixa', 
        'Discrepancia_Notas_IDA'
    ]
    
    logger.info(f"Features criadas com sucesso: {novas_colunas}")
    return df_feat

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Orquestrador do módulo de Feature Engineering."""
    df_final = create_academic_features(df)
    return df_final