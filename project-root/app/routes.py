from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

# Carregamento do modelo em cache na inicialização do servidor.
# Isso evita lentidão, pois não carrega o modelo do disco a cada requisição.
MODEL_PATH = "model/modelo_defasagem.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Modelo preditivo carregado com sucesso.")
except FileNotFoundError:
    logger.error(f"Arquivo de modelo não encontrado em {MODEL_PATH}.")
    model_pipeline = None

# Schema de dados esperado na requisição JSON
# Campos opcionais (None) serão preenchidos pela pipeline do scikit-learn (mediana/constante)
class StudentData(BaseModel):
    Fase: str = Field(..., description="Fase escolar do aluno (ex: Fase 1 (3º e 4º ano))")
    Idade: float = Field(..., description="Idade atual do aluno")
    Gênero: str = Field(..., description="Feminino ou Masculino")
    IAA: Optional[float] = Field(default=None, description="Indicador de Auto Avaliação")
    IEG: Optional[float] = Field(default=None, description="Indicador de Engajamento")
    IPS: Optional[float] = Field(default=None, description="Indicador Psicossocial")
    IDA: Optional[float] = Field(default=None, description="Indicador de Aprendizagem")
    IPP: Optional[float] = Field(default=None, description="Indicador Psicopedagógico")
    IPV: Optional[float] = Field(default=None, description="Indicador de Ponto de Virada")
    IAN: Optional[float] = Field(default=None, description="Indicador de Adequação Nivelamento")
    Mat: Optional[float] = Field(default=None, description="Nota de Matemática")
    Por: Optional[float] = Field(default=None, description="Nota de Português")
    Ing: Optional[float] = Field(default=None, description="Nota de Inglês")

@router.post("/predict", tags=["Predição"])
def predict_risk(student: StudentData):
    """
    Recebe os dados de um aluno e retorna a probabilidade e a classe de risco de defasagem.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="O modelo preditivo não está disponível no servidor.")
    
    try:
        # Pydantic v2: model_dump() converte o schema para dicionário
        # Transformamos em um DataFrame de 1 linha, formato exigido pelo scikit-learn
        df_input = pd.DataFrame([student.model_dump()])
        
        # O modelo faz o pré-processamento (Pipeline) e a predição internamente
        prediction = model_pipeline.predict(df_input)[0]
        
        # predict_proba retorna probabilidade [Classe 0, Classe 1]
        probability = model_pipeline.predict_proba(df_input)[0][1] 
        
        # Formatação profissional da resposta
        risco_bool = bool(prediction == 1)
        
        return {
            "risco_defasagem": risco_bool,
            "probabilidade_risco": round(float(probability), 4),
            "mensagem_alerta": "ALTO risco de defasagem." if risco_bool else "BAIXO risco de defasagem.",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Erro durante a predição: {e}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar os dados: {str(e)}")