FROM qdrant/qdrant:latest

# Expor a porta que o QDrant usa
EXPOSE 6333
EXPOSE 6334

# Definir o volume de persistência
VOLUME /qdrant/storage

# Definir diretório de trabalho
WORKDIR /qdrant

# Comando para executar o QDrant quando o contêiner for iniciado
CMD ["./qdrant"]