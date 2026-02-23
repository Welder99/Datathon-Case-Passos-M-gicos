from fastapi import FastAPI
from app.routes import router

# Inicialização da aplicação com metadados para a documentação automática (Swagger)
app = FastAPI(
    title="API Passos Mágicos - Risco de Defasagem",
    description="API preditiva para estimar o risco de defasagem escolar de estudantes da Associação Passos Mágicos.",
    version="1.0.0"
)

# Inclui as rotas separadas no arquivo routes.py
app.include_router(router)

@app.get("/", tags=["Health Check"])
def health_check():
    """Endpoint básico para verificar se a API está no ar."""
    return {"status": "ok", "message": "API Passos Mágicos operante."}