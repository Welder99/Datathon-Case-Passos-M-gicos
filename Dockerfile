# Usa a imagem oficial do Python 3.12 na versão slim (mais leve e segura para produção)
FROM python:3.12-slim

# Define metadados da imagem
LABEL maintainer="Seu Nome <seu.email@exemplo.com>"
LABEL description="API de Machine Learning para predição de risco de defasagem escolar - Passos Mágicos"

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de requisitos primeiro
# Isso otimiza o cache do Docker. Se o código mudar, mas os requisitos não, 
# o Docker não precisa baixar todas as bibliotecas de novo.
COPY requirements.txt .

# Instala as dependências do Python
# A flag --no-cache-dir mantém a imagem final menor, não guardando arquivos temporários do pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o diretório de trabalho (/app)
COPY . .

# Expõe a porta 8000, que é a porta padrão que o Uvicorn vai usar
EXPOSE 8000

# Comando para iniciar o servidor da API usando o Uvicorn
# 0.0.0.0 garante que o servidor aceite conexões externas de fora do container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]