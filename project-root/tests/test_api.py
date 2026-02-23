import pytest
from fastapi.testclient import TestClient
from app.main import app

# Instancia o cliente de teste do FastAPI
client = TestClient(app)

def test_health_check():
    """Testa se a API está no ar."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API Passos Mágicos operante."}

def test_predict_endpoint_success():
    """Testa uma requisição válida para o endpoint de predição."""
    
    # Payload simulando o input do Pydantic (StudentData)
    payload = {
        "Fase": "Fase 1 (3º e 4º ano)",
        "Idade": 12,
        "Gênero": "Feminino",
        "IAA": 8.5,
        "IEG": 9.0,
        "Mat": 7.5
        # Deixando colunas de fora propositalmente para testar a imputação de nulos
    }
    
    response = client.post("/predict", json=payload)
    
    # Se o modelo não estiver treinado localmente ao rodar o teste, ele vai dar erro 500.
    # Vamos tratar isso no teste para garantir que a rota existe e o schema é válido.
    if response.status_code == 200:
        data = response.json()
        assert "risco_defasagem" in data
        assert "probabilidade_risco" in data
        assert "status" in data
        assert data["status"] == "success"
    else:
        # Se o modelo .joblib não existir, a API deve retornar 500 informando isso
        assert response.status_code == 500
        assert "não está disponível" in response.json()["detail"]

def test_predict_endpoint_validation_error():
    """Testa se o Pydantic bloqueia requisições com dados obrigatórios faltando."""
    
    # Faltam campos obrigatórios como 'Fase' e 'Gênero'
    payload = {
        "Idade": 15
    }
    
    response = client.post("/predict", json=payload)
    
    # 422 é o erro padrão do FastAPI/Pydantic para falha de validação (Unprocessable Entity)
    assert response.status_code == 422