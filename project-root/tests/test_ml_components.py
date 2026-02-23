import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline

from src.feature_engineering import run_feature_engineering
from src.evaluate import evaluate_predictions, save_metrics
from src.train import create_pipeline, train_and_evaluate_model
from src.utils import setup_logger, save_model, load_model
from src.preprocessing import load_and_standardize_data, run_preprocessing_pipeline
from src.monitor import generate_drift_report

# ==========================================
# 1. Testes de Feature Engineering
# ==========================================
def test_create_academic_features():
    df_mock = pd.DataFrame({
        'Mat': [10.0, 4.0], 'Por': [9.0, 5.0], 'Ing': [8.0, 3.0],
        'IAA': [9.0, 4.0], 'IEG': [8.5, 3.5], 'IPS': [9.0, 4.0],
        'IDA': [8.0, 4.0], 'IPP': [9.0, 4.0], 'IPV': [8.0, 4.0], 'IAN': [9.0, 3.0]
    })
    
    df_result = run_feature_engineering(df_mock)
    
    assert 'Media_Notas' in df_result.columns
    assert 'Alerta_Engajamento_Baixo' in df_result.columns
    assert df_result['Alerta_Engajamento_Baixo'].iloc[1] == 1
    assert df_result['Alerta_Engajamento_Baixo'].iloc[0] == 0

# ==========================================
# 2. Testes de Avaliação (Evaluate)
# ==========================================
def test_evaluate_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0] 
    y_proba = [0.9, 0.1, 0.4, 0.2]
    
    metrics = evaluate_predictions(y_true, y_pred, y_proba, return_dict=True)
    assert metrics['recall_risk_class'] == 0.5  

@patch('src.evaluate.json.dump')
@patch('src.evaluate.open')
@patch('src.evaluate.os.makedirs')
def test_save_metrics(mock_makedirs, mock_open, mock_json_dump):
    save_metrics({'acc': 1.0}, "pasta_teste/dummy_path.json")
    mock_open.assert_called_once()
    mock_makedirs.assert_called_once()

# ==========================================
# 3. Testes de Treinamento (Pipeline)
# ==========================================
def test_create_pipeline():
    pipeline = create_pipeline(['Idade', 'Mat'], ['Fase', 'Gênero'])
    assert isinstance(pipeline, Pipeline)
    assert 'preprocessor' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps

@patch('src.train.os.makedirs') # Protege contra o erro de diretório no Windows
@patch('src.train.joblib.dump')
@patch('src.train.pd.read_csv')
def test_train_and_evaluate_model(mock_read_csv, mock_joblib_dump, mock_makedirs):
    # Simulando um target com ambas as classes para evitar o aviso ROC AUC
    mock_df = pd.DataFrame({
        'RA': ['1', '2', '3', '4'], 'Ano_Ref': [2022, 2023, 2024, 2024],
        'Risco_Defasagem': [1, 0, 1, 0], 'Idade': [12, 13, 14, 15],
        'Mat': [8.0, 9.0, 7.0, 6.0], 'Fase': ['F1', 'F2', 'F3', 'F4'], 'Gênero': ['M', 'F', 'M', 'F']
    })
    mock_read_csv.return_value = mock_df
    train_and_evaluate_model("pasta/falso.csv", "pasta/falso.joblib")
    mock_joblib_dump.assert_called_once()

# ==========================================
# 4. Testes de Utils
# ==========================================
def test_setup_logger():
    logger = setup_logger('test_logger')
    assert logger.name == 'test_logger'

@patch('src.utils.os.makedirs')
@patch('src.utils.joblib.dump')
def test_save_model(mock_joblib_dump, mock_makedirs):
    save_model("modelo", "pasta/caminho.joblib")
    mock_joblib_dump.assert_called_once()

@patch('src.utils.joblib.load')
@patch('src.utils.os.path.exists', return_value=True)
def test_load_model(mock_exists, mock_joblib_load):
    load_model("pasta/caminho.joblib")
    mock_joblib_load.assert_called_once()

# ==========================================
# 5. Testes de Preprocessing
# ==========================================
@patch('pandas.read_excel')
def test_load_and_standardize_data(mock_read_excel):
    mock_df_2022 = pd.DataFrame({
        'RA': ['123'], 'Defasagem': [-1], 'Matem': [10]
    })
    mock_read_excel.return_value = {'aba_2022': mock_df_2022}
    
    df_result = load_and_standardize_data('pasta/caminho_falso.xlsx')
    assert 'Defasagem' in df_result.columns
    assert 'Ano_Ref' in df_result.columns

@patch('src.preprocessing.os.makedirs')
@patch('src.preprocessing.load_and_standardize_data')
@patch('src.preprocessing.pd.DataFrame.to_csv')
def test_run_preprocessing_pipeline(mock_to_csv, mock_load, mock_makedirs):
    mock_load.return_value = pd.DataFrame({
        'RA': ['123'], 'Defasagem': [-1], 'Mat': [10]
    })
    df = run_preprocessing_pipeline('falso.xlsx', 'pasta/falso.csv')
    assert 'Risco_Defasagem' in df.columns
    mock_to_csv.assert_called_once()

# ==========================================
# 6. Testes de Monitoramento (Drift)
# ==========================================
@patch('src.monitor.os.makedirs')
@patch('src.monitor.pd.read_csv')
def test_generate_drift_report(mock_read_csv, mock_makedirs):
    """
    Testa o script de drift mockando a biblioteca Evidently para evitar 
    que o pytest falhe se a biblioteca não estiver 100% instalada no ambiente.
    """
    mock_df = pd.DataFrame({'Mat': [10, 9], 'RA': ['1', '2'], 'Risco_Defasagem': [1, 0]})
    mock_read_csv.return_value = mock_df
    
    with patch.dict('sys.modules', {'evidently.report': MagicMock(), 'evidently.metric_preset': MagicMock()}):
        generate_drift_report("pasta/ref.csv", "pasta/curr.csv", "pasta/saida.html")
    
    # Valida se os CSVs foram lidos
    assert mock_read_csv.call_count == 2