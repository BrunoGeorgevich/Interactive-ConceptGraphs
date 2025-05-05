# QDrant Vector Database

Este diretório contém arquivos para implantar um banco de dados vetorial QDrant usando Docker.

## Estrutura de Arquivos

- `Dockerfile`: Define a imagem do QDrant
- `docker-compose.yml`: Configura o serviço do QDrant
- `config/config.yaml`: Arquivo de configurações do QDrant

## Como Executar

Para iniciar o serviço QDrant:

```bash
cd docker
docker-compose up -d
```

Após a execução, o serviço estará disponível em:
- API REST: http://localhost:6333
- Interface Web: http://localhost:6334

## Como Parar

```bash
docker-compose down
```

Para parar e remover os volumes (CUIDADO: isso irá apagar todos os dados):

```bash
docker-compose down -v
```

## Informações Adicionais

- Os dados persistentes são armazenados no volume Docker `qdrant_data`
- As configurações podem ser ajustadas no arquivo `config/config.yaml`