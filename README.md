🚀 Datathon: Case Passos Mágicos - Risco de Defasagem Escolar
1. Visão Geral do Projeto
Objetivo: Este projeto faz parte da conclusão da Pós-Graduação em Machine Learning Engineering. O objetivo é desenvolver um modelo preditivo capaz de estimar o risco de defasagem escolar de cada estudante da Associação Passos Mágicos, utilizando dados educacionais reais dos períodos de 2022, 2023 e 2024.
+1


Solução Proposta: Foi construída uma pipeline completa de Machine Learning, aplicando as melhores práticas de MLOps. A solução abrange desde o pré-processamento de dados até o deploy de uma API robusta, garantindo escalabilidade e monitoramento contínuo.
+2

Stack Tecnológica:


Linguagem: Python 3.12 


Bibliotecas ML: scikit-learn, pandas, numpy 


API: FastAPI 


Serialização: joblib 


Testes Automatizados: pytest & pytest-cov 
+1


Empacotamento: Docker 


Monitoramento: Evidently AI & Logging 
+1

2. Estrutura do Projeto 

Plaintext
project-root/
├── app/                      # Código-fonte da API [cite: 54]
│   ├── main.py               # Inicialização do FastAPI
│   └── routes.py             # Definição de endpoints e schemas
├── data/                     # Armazenamento da base de dados bruta e processada
├── model/                    # Modelos serializados (.joblib) [cite: 54]
├── notebooks/                # Jupyter Notebooks para análise exploratória (EDA) [cite: 54]
├── src/                      # Pipeline de Machine Learning [cite: 54]
│   ├── preprocessing.py      # Limpeza e padronização (Multi-aba Excel)
│   ├── feature_engineering.py# Criação de atributos acadêmicos
│   ├── train.py              # Script de treinamento e validação
│   ├── evaluate.py           # Cálculo de métricas e relatórios
│   ├── utils.py              # Funções auxiliares e loggers
│   └── monitor.py            # Geração de relatórios de Data Drift
├── tests/                    # Testes unitários (Mínimo 80% de cobertura) [cite: 34, 54]
├── Dockerfile                # Configuração para empacotamento da aplicação [cite: 29, 54]
├── requirements.txt          # Dependências do projeto [cite: 54, 60]
└── README.md                 # Documentação principal
3. Instruções de Execução e Deploy 

Pré-requisitos:

Python 3.12 instalado ou Docker.

Instalação e Treinamento:
Criar ambiente virtual: python -m venv venv

Ativar venv: venv\Scripts\activate (Windows) ou source venv/bin/activate (Linux/Mac)


Instalar dependências: pip install -r requirements.txt 

Processar dados e Treinar:


python src/preprocessing.py 


python src/train.py 

Execução via Docker:
+1

Build da imagem:

Bash
docker build -t passos-magicos-api .
Rodar o container:

Bash
docker run -d -p 8000:8000 --name api-pm passos-magicos-api
4. Testes e Qualidade 

Para garantir a confiabilidade do modelo e do código, foram implementados testes unitários com foco em lógica de negócio e processamento de dados.

Comando: pytest --cov=app --cov=src --cov-report=term-missing

Resultado Obtido: 89% de cobertura de código.

5. Exemplo de Chamada à API 

A API fornece documentação automática via Swagger em http://localhost:8000/docs.


Requisição (POST /predict):

JSON
{
  "Fase": "Fase 1 (3º e 4º ano)",
  "Idade": 12.0,
  "Gênero": "Masculino",
  "IAA": 8.5,
  "IEG": 9.0,
  "Mat": 7.0,
  "Por": 8.0
}

Resposta:

JSON
{
  "risco_defasagem": false,
  "probabilidade_risco": 0.1245,
  "mensagem_alerta": "BAIXO risco de defasagem.",
  "status": "success"
}
6. Pipeline de Machine Learning 


Pré-processamento: Unificação de abas históricas, padronização de nomenclatura de colunas e criação do target binário (Risco_Defasagem).


Engenharia de Features: Cálculo de médias acadêmicas, indicadores comportamentais e detecção de discrepâncias entre notas e indicadores IDA.

Treinamento: Divisão temporal (Treino: 2022/2023; Teste: 2024). Uso de Random Forest com pesos balanceados para otimização de Recall.


Monitoramento: Integração com Evidently AI para detecção de Data Drift, garantindo que o modelo seja recalibrado caso os padrões dos alunos mudem drasticamente