import pandas as pd
import numpy as np
import logging
import os

# Configuração básica de log para monitoramento em produção
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_standardize_data(filepath: str) -> pd.DataFrame:
    """
    Carrega todas as abas da base de dados Excel (.xlsx) e padroniza as colunas.
    
    Args:
        filepath (str): Caminho para o arquivo Excel.
        
    Returns:
        pd.DataFrame: DataFrame único com todas as abas concatenadas e padronizadas.
    """
    logger.info(f"Iniciando o carregamento da base de dados Excel: {filepath}")
    
    try:
        # sheet_name=None lê todas as abas do Excel de uma vez.
        # Retorna um dicionário: {'NomeDaAba': DataFrame}
        sheets = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado em {filepath}. Certifique-se de que a pasta 'data/' existe.")
        raise
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo Excel: {e}")
        raise

    df_list = []
    
    for sheet_name, df in sheets.items():
        logger.info(f"Processando a aba: {sheet_name} | Linhas originais: {len(df)}")
        
        # 1. Identifica o ano da aba para usarmos no split temporal depois
        ano_ref = 2024 # Padrão caso não ache no nome
        if '2022' in sheet_name: ano_ref = 2022
        elif '2023' in sheet_name: ano_ref = 2023
        elif '2024' in sheet_name: ano_ref = 2024
        
        df['Ano_Ref'] = ano_ref

        # 2. Padronização de nomes de colunas (pois mudam de 2022 para 2023/2024)
        if 'Defas' in df.columns and 'Defasagem' not in df.columns:
            df.rename(columns={'Defas': 'Defasagem'}, inplace=True)
            
        rename_dict = {
            'Fase ideal': 'Fase Ideal',
            'Matem': 'Mat',
            'Portug': 'Por',
            'Inglês': 'Ing',
            'Ano nasc': 'Data de Nasc', 
            'Idade 22': 'Idade'
        }
        # Renomeia apenas as colunas que existirem nesta aba específica
        df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
        
        # 3. Filtro de colunas de interesse
        cols_desejadas = [
            'RA', 'Ano_Ref', 'Fase', 'Idade', 'Gênero', 
            'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN', 
            'Mat', 'Por', 'Ing', 'Defasagem'
        ]
        
        cols_existentes = [col for col in cols_desejadas if col in df.columns]
        df_filtrado = df[cols_existentes].copy()
        
        # O IPP não existia em 2022. Adicionamos como NaN para manter a simetria ao concatenar.
        if 'IPP' not in df_filtrado.columns:
            df_filtrado['IPP'] = np.nan
            
        df_list.append(df_filtrado)
        
    # 4. Concatena todas as abas processadas
    df_final = pd.concat(df_list, ignore_index=True)
    logger.info(f"Todas as abas foram concatenadas. Shape final: {df_final.shape}")
    
    return df_final

def prepare_target_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza a limpeza de tipos e cria a variável alvo (Target)."""
    logger.info("Iniciando a preparação do Target e conversão de tipos...")
    
    if 'Defasagem' not in df.columns:
        raise KeyError("A coluna 'Defasagem' não foi encontrada. Verifique o arquivo Excel.")
        
    df = df.dropna(subset=['Defasagem']).copy()
    
    # Risco (1): Defasagem negativa. Sem Risco (0): Defasagem 0 ou positiva.
    df['Risco_Defasagem'] = df['Defasagem'].apply(lambda x: 1 if float(x) < 0 else 0)
    
    # Conversão de numéricos
    num_cols = ['Idade', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN', 'Mat', 'Por', 'Ing']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Remove a coluna original para evitar Data Leakage
    df = df.drop(columns=['Defasagem'])
    
    logger.info(f"Preparação concluída. Distribuição do Target:\n{df['Risco_Defasagem'].value_counts()}")
    return df

def run_preprocessing_pipeline(filepath: str, output_path: str = None) -> pd.DataFrame:
    """Orquestrador do módulo de pré-processamento."""
    df_raw = load_and_standardize_data(filepath)
    df_clean = prepare_target_and_types(df_raw)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Salvaremos o dado limpo como CSV para facilitar a leitura rápida na etapa de treino
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Dados salvos em {output_path}")
        
    return df_clean

if __name__ == "__main__":
    # Apontando diretamente para o arquivo Excel dentro da pasta data/
    CAMINHO_EXCEL = 'data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
    CAMINHO_SAIDA = 'data/dados_limpos.csv'
    
    df_processado = run_preprocessing_pipeline(filepath=CAMINHO_EXCEL, output_path=CAMINHO_SAIDA)