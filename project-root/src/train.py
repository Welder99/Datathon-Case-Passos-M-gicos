import pandas as pd
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# Configuração de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_pipeline(num_features: list, cat_features: list) -> Pipeline:
    """
    Cria a pipeline de Machine Learning contendo pré-processamento e o classificador.
    """
    logger.info("Construindo a pipeline de pré-processamento...")
    
    # Pipeline para variáveis numéricas: Imputação com a mediana e padronização (Z-score)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variáveis categóricas: Imputação de valor constante e One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combina os processadores numéricos e categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # Adiciona o modelo final à pipeline. 
    # Usamos RandomForest por ser robusto a outliers e lidar bem com relações não-lineares.
    # class_weight='balanced' é crucial para lidar com possível desbalanceamento do target.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    
    return model_pipeline

def train_and_evaluate_model(data_path: str, model_save_path: str):
    """
    Orquestra o carregamento dos dados, split temporal, treinamento, avaliação e salvamento do modelo.
    """
    logger.info(f"Carregando dados limpos de: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Split Temporal: Treinar com o passado (2022/2023) e testar com o futuro (2024)
    # Isso simula o cenário real de produção de forma muito mais fiel do que um split aleatório.
    train_df = df[df['Ano_Ref'].isin([2022, 2023])]
    test_df = df[df['Ano_Ref'] == 2024]
    
    if test_df.empty:
        logger.warning("Dados de 2024 não encontrados. Realizando split aleatório (80/20).")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Risco_Defasagem'])
    
    logger.info(f"Tamanho do Treino: {len(train_df)} | Tamanho do Teste: {len(test_df)}")
    
    # Definindo as features
    target = 'Risco_Defasagem'
    drop_cols = ['RA', 'Ano_Ref', target] # Colunas que não são features
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]
    
    # Separando numericas e categoricas
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Criar e Treinar a Pipeline
    pipeline = create_pipeline(num_features, cat_features)
    
    logger.info("Iniciando o treinamento do modelo (Random Forest)...")
    pipeline.fit(X_train, y_train)
    logger.info("Treinamento concluído com sucesso.")
    
    # 3. Avaliação do Modelo
    logger.info("Avaliando o modelo na base de teste...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Métricas
    # A métrica Recall para a classe 1 é a mais importante, pois queremos mitigar o erro de 
    # não identificar um aluno em risco de defasagem escolar.
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO (Foco em Produção)")
    print("="*50)
    print(report)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"F1-Score (Classe Positiva): {f1:.4f}")
    print("="*50 + "\n")
    
    # 4. Serialização do Modelo
    # Utilizando joblib conforme exigido, excelente para pipelines com arrays numpy
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    logger.info(f"Modelo serializado com sucesso e salvo em: {model_save_path}")

if __name__ == "__main__":
    # Caminho do dataset limpo (gerado pelo preprocessing.py)
    DADOS_LIMPOS_PATH = "C:/Users/welde/OneDrive/Área de Trabalho/TC5/project-root/data/dados_limpos.csv" 
    
    # Caminho onde o modelo empacotado será salvo
    MODELO_SAVE_PATH = "model/modelo_defasagem.joblib"
    
    train_and_evaluate_model(DADOS_LIMPOS_PATH, MODELO_SAVE_PATH)