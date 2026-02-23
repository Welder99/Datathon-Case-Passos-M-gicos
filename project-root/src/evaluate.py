import pandas as pd
import logging
import json
import os
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    f1_score, 
    recall_score, 
    precision_score, 
    confusion_matrix
)

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true, y_pred, y_proba=None, return_dict=False):
    """
    Avalia as predições do modelo de risco de defasagem.
    Foca principalmente no Recall da classe 1 (Risco), pois o custo de um Falso Negativo
    (não identificar um aluno em risco) é o mais prejudicial para a Passos Mágicos.
    
    Args:
        y_true: Valores reais (target).
        y_pred: Valores preditos pelo modelo.
        y_proba: Probabilidades preditas (necessário para ROC-AUC).
        return_dict (bool): Se True, retorna as métricas como dicionário (útil para logs/MLflow).
    """
    logger.info("Iniciando avaliação do modelo...")
    
    metrics = {
        "accuracy": None,
        "f1_score_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_score_risk_class": f1_score(y_true, y_pred, pos_label=1),
        "recall_risk_class": recall_score(y_true, y_pred, pos_label=1),
        "precision_risk_class": precision_score(y_true, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None
    }
    
    # Gerando o relatório detalhado do Scikit-Learn
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print("📈 RELATÓRIO DE AVALIAÇÃO DO MODELO - PASSOS MÁGICOS")
    print("="*60)
    print(report)
    print("-" * 60)
    print("Matriz de Confusão:")
    print(f"Verdadeiros Negativos (Sem Risco): {conf_matrix[0][0]} | Falsos Positivos (Alarme Falso): {conf_matrix[0][1]}")
    print(f"Falsos Negativos (Risco Oculto): {conf_matrix[1][0]} | Verdadeiros Positivos (Risco Detectado): {conf_matrix[1][1]}")
    print("-" * 60)
    
    if metrics["roc_auc"]:
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Recall da Classe de Risco (1): {metrics['recall_risk_class']:.4f}")
    print("="*60 + "\n")
    
    if return_dict:
        return metrics

def save_metrics(metrics: dict, filepath: str = "model/metrics.json"):
    """Salva as métricas em um arquivo JSON para histórico."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métricas salvas com sucesso em: {filepath}")