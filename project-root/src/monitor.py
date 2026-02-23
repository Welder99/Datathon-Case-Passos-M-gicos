import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def generate_drift_report(reference_data_path: str, current_data_path: str, output_html: str = "monitoring/drift_dashboard.html"):
    """
    Gera um relatório de Data Drift comparando os dados de treinamento (referência)
    com os dados mais recentes de produção (current).
    """
    logger.info("Iniciando análise de Data Drift...")
    
    # IMPORTAÇÃO TARDIA: Protege os testes e evita quebra caso a lib falhe no servidor
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as e:
        logger.error(f"Evidently não encontrado ou com erro: {e}. O dashboard não será gerado.")
        return
        
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    
    df_ref = pd.read_csv(reference_data_path)
    df_curr = pd.read_csv(current_data_path)
    
    cols_to_drop = ['RA', 'Ano_Ref', 'Risco_Defasagem']
    df_ref = df_ref.drop(columns=[c for c in cols_to_drop if c in df_ref.columns])
    df_curr = df_curr.drop(columns=[c for c in cols_to_drop if c in df_curr.columns])
    
    drift_report = Report(metrics=[DataDriftPreset()])
    
    logger.info("Calculando métricas de Drift...")
    drift_report.run(reference_data=df_ref, current_data=df_curr)
    
    drift_report.save_html(output_html)
    logger.info(f"Dashboard de Drift gerado com sucesso em: {output_html}")

if __name__ == "__main__":
    pass