import pandas as pd
import pytest
from src.preprocessing import prepare_target_and_types

@pytest.fixture
def sample_raw_data():
    """Cria um DataFrame simulado para os testes."""
    data = {
        'RA': ['RA-1', 'RA-2', 'RA-3'],
        'Fase': ['Fase 1', 'Fase 2', 'Fase 3'],
        'Idade': [10, '12', 15], # Simulando sujeira ('12' como string)
        'Defasagem': [-1.0, 0.0, 2.0], # -1 = Risco (1), 0 e 2 = Sem Risco (0)
        'IAA': [8.0, 9.0, 'NaN'],
        'IEG': [7.5, 8.5, 6.0],
        'IPS': [9.0, 9.0, 9.0],
        'IDA': [5.0, 6.0, 7.0],
        'IPP': [7.0, 8.0, 9.0],
        'IPV': [6.0, 7.0, 8.0],
        'IAN': [5.0, 5.0, 5.0],
        'Mat': [8.0, 9.0, 10.0],
        'Por': [7.0, 8.0, 9.0],
        'Ing': [6.0, 7.0, 8.0]
    }
    return pd.DataFrame(data)

def test_prepare_target_and_types(sample_raw_data):
    """Testa se a variável alvo é criada corretamente e a defasagem original é removida."""
    
    df_clean = prepare_target_and_types(sample_raw_data)
    
    # Verifica se a coluna alvo foi criada
    assert 'Risco_Defasagem' in df_clean.columns
    
    # Verifica se a regra de negócio do risco foi aplicada corretamente
    # RA-1 tinha Defasagem -1 (deve ser 1)
    assert df_clean.loc[df_clean['RA'] == 'RA-1', 'Risco_Defasagem'].values[0] == 1
    # RA-2 tinha Defasagem 0 (deve ser 0)
    assert df_clean.loc[df_clean['RA'] == 'RA-2', 'Risco_Defasagem'].values[0] == 0
    # RA-3 tinha Defasagem 2 (deve ser 0)
    assert df_clean.loc[df_clean['RA'] == 'RA-3', 'Risco_Defasagem'].values[0] == 0
    
    # Verifica a conversão de tipos (Idade era string em um dos casos)
    assert pd.api.types.is_numeric_dtype(df_clean['Idade'])
    
    # Verifica se a coluna original foi removida para evitar data leakage
    assert 'Defasagem' not in df_clean.columns