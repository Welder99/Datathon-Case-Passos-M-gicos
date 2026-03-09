# 1. Imagem base oficial do Python (Slim para performance e segurança)
FROM python:3.12-slim

# 2. Configurações de ambiente para evitar arquivos .pyc e garantir logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 3. Define o diretório de trabalho raiz
WORKDIR /app

# 4. Instala dependências de compilação (necessárias para algumas libs de ML)
# Limpa o cache do apt para manter a imagem leve
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copia e instala as dependências do projeto (Aproveita o Cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copia todo o código-fonte para o container
# Certifique-se de estar na pasta onde estão 'app/', 'src/', 'model/' e 'data/'
COPY . .

# 7. Cria um usuário não-root por segurança (Boa prática de produção)
RUN useradd -m myuser
USER myuser

# 8. Expõe a porta padrão do FastAPI
EXPOSE 8000

# 9. Comando de inicialização Robusto
# O uvicorn é chamado como módulo do python para garantir o contexto do diretório atual
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]