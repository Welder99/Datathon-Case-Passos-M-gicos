🚀 Datathon: Case Passos Mágicos - Risco de Defasagem Escolar
🔗 Link da API em Produção
Acesse a documentação interativa (Swagger) da solução hospedada:
👉 https://datathon-case-passos-m-gicos.onrender.com/docs

1. Visão Geral do Projeto
Objetivo: Este projeto foi desenvolvido para a conclusão da Pós-Graduação em Machine Learning Engineering. O foco é fornecer à Associação Passos Mágicos uma ferramenta preditiva capaz de identificar estudantes em risco de defasagem escolar, utilizando dados históricos de 2022, 2023 e 2024.

Solução Proposta: Uma pipeline de MLOps de ponta a ponta que transforma dados brutos em inteligência acionável. A solução prioriza o Recall, garantindo que o máximo de alunos em risco seja identificado para intervenção pedagógica precoce.

Stack Tecnológica:
Linguagem: Python 3.12

Machine Learning: scikit-learn, pandas, numpy, joblib

API & Servidor: FastAPI, Uvicorn

Qualidade: Pytest, Pytest-cov (89% de cobertura)

DevOps: Docker, Render (Cloud Deploy)

Monitoramento: Evidently AI (Análise de Data Drift)

2. Estrutura do Projeto
Plaintext
project-root/
├── app/                      # Camada de entrega: API FastAPI
│   ├── main.py               # Ponto de entrada do servidor
│   └── routes.py             # Endpoints, Schemas e lógica de predição
├── data/                     # Dados brutos (Excel) e processados (CSV)
├── model/                    # Modelos serializados (.joblib)
├── notebooks/                # Análise Exploratória (EDA) e Insights
├── src/                      # Core da Pipeline de ML
│   ├── preprocessing.py      # Padronização de dados históricos
│   ├── feature_engineering.py# Criação de indicadores de engajamento
│   ├── train.py              # Treino com Divisão Temporal
│   ├── evaluate.py           # Métricas de performance
│   ├── monitor.py            # Monitoramento de Drift
│   └── utils.py              # Loggers e funções auxiliares
├── tests/                    # Testes automatizados (Unitários e Integração)
├── Dockerfile                # Empacotamento profissional (Multi-layer)
└── requirements.txt          # Gestão de dependências
3. Instruções de Execução
Localmente (Ambiente Virtual)
python -m venv venv

source venv/bin/activate (Linux/Mac) ou .\venv\Scripts\activate (Windows)

pip install -r requirements.txt

python src/train.py (Para gerar o modelo)

python -m uvicorn app.main:app --reload

Via Docker (Recomendado para Avaliação)
A solução está configurada com PYTHONPATH dinâmico para evitar erros de importação:

Bash
# Build da imagem
docker build -t passos-magicos-api .

# Execução do container
docker run -d -p 8000:8000 --name api-pm passos-magicos-api
Acesse: http://localhost:8000/docs

4. Testes e Qualidade
O projeto adota uma postura rigorosa de QA, garantindo que alterações na pipeline não degradem o modelo ou a API.

Comando: python -m pytest --cov=app --cov=src --cov-report=term-missing

Resultado: 89% de cobertura de código.

5. Pipeline e Estratégia de ML
Pré-processamento: Tratamento de dados de múltiplas abas Excel, normalização de nomes de colunas e criação da variável alvo baseada em anos de defasagem.

Engenharia de Features: Criação de colunas sintéticas como Media_Notas e Alerta_Engajamento, cruzando dados acadêmicos com indicadores comportamentais (IEG/IDA).

Seleção de Modelo: Random Forest Classifier com pesos balanceados.

Justificativa da Métrica: O modelo foi otimizado para Recall. No contexto social, o "Falso Negativo" (não detectar um aluno que precisa de ajuda) é mais prejudicial que o "Falso Positivo".

6. Exemplo de Uso (API)
Endpoint: POST /predict

Input:

JSON
{
  "Fase": "Fase 2",
  "Idade": 14.0,
  "Gênero": "Feminino",
  "IAA": 7.5,
  "IEG": 4.0,
  "Mat": 5.5,
  "Por": 6.0
}
Output:

JSON
{
  "risco_defasagem": true,
  "probabilidade_risco": 0.82,
  "mensagem_alerta": "ALTO risco detectado. Recomenda-se atenção pedagógica.",
  "status": "success"
}