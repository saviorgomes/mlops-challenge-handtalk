# 🌐 MLOps Challenge Starter — Neural Machine Translation (EN → PT)

Repositório inicial (**starter kit**) para o desafio de MLOps. Contém as rotinas base de preparação de dados, treinamento e inferência para um modelo de tradução automática Inglês -> Português, utilizando um **Transformer** customizado com TensorFlow/Keras.

> **O objetivo do candidato é construir a automação end-to-end (CI/CD, orquestração, monitoramento, etc.) em cima destas rotinas já fornecidas.**
>
> 📄 **Consulte a especificação completa do desafio em [CHALLENGE.md](CHALLENGE.md).**

---

## 📋 Visão Geral

Este repositório fornece as **rotinas base** que o candidato utilizará para implementar a automação MLOps. As peças já inclusas são:

| Componente | Descrição |
|---|---|
| **Preparação de Dados** | Download e tokenização do dataset [ParaCrawl EN-PT](https://www.paracrawl.eu/) via TensorFlow Datasets, exportando TFRecords prontos para treino |
| **Treinamento** | Modelo Transformer (encoder-decoder) com warmup schedule, masked loss/accuracy e versionamento automático de artefatos |
| **Inference API** | API REST (FastAPI + Uvicorn) para tradução em tempo real, com métricas, health check e hot-reload de modelos |
| **Testes** | Suite de testes de contrato da API via Pytest + HTTPX |

> [!IMPORTANT]
> Estas rotinas já funcionam individualmente via Docker Compose. O desafio consiste em **automatizar e orquestrar** todo o ciclo de vida do modelo de ponta a ponta.

---

## 🧭 Resumo Executivo

Esta entrega prioriza os pontos centrais do desafio: **automação do ciclo de vida do modelo**, **operação da plataforma** e **observabilidade do ambiente**, preservando as rotinas-base de ML já fornecidas.

Os principais objetivos atendidos foram:

- Disparo único do pipeline via `n8n`, com fluxo de **preparar → treinar → validar → publicar → deploy**
- Proteção da API com **gateway**, autenticação, rate limiting e logging
- Pipeline de **CI/CD** com lint, testes, build e publicação de imagem
- Observabilidade com **métricas, logs, health checks e alertas**
- Rastreabilidade de artefatos com `run_id`, `git_sha`, métricas e metadados

Decisão importante de implementação:

- A camada de automação foi mantida **fora dos scripts-base de ML**, para respeitar a orientação do desafio de não alterar o comportamento das rotinas de treinamento e preparação apenas para “embutir MLOps”

---

## 🏗️ Arquitetura

```
mlops_challenge_starter/
├── ml/                        # Pipeline de ML
│   ├── common.py              # Utilitários (I/O JSON, hashing, run_id)
│   ├── tokenizers.py          # Download e carregamento de tokenizers
│   ├── prepare_dataset.py     # Preparação do dataset (TFRecords)
│   ├── model.py               # Transformer (Encoder/Decoder/Positional Encoding)
│   └── train.py               # Loop de treino, exportação de SavedModel
│
├── inference_api/             # API de inferência
│   ├── main.py                # Endpoints FastAPI
│   ├── model_manager.py       # Gerenciamento thread-safe do SavedModel
│   ├── schemas.py             # Schemas Pydantic (request/response)
│   ├── metrics.py             # Contadores de métricas da aplicação
│   └── logging_config.py      # Configuração de logging estruturado
│
├── tests/                     # Testes automatizados
│   └── test_api_contract.py   # Testes de contrato dos endpoints
│
├── data/                      # Dados processados (gerados)
├── artifacts/                 # Artefatos de treino (SavedModel, configs)
├── Dockerfile                 # Imagem base (Python 3.11-slim)
├── docker-compose.yml         # Orquestração de todos os serviços
└── requirements.txt           # Dependências Python
```

---

## 🧠 Modelo

O modelo é um **Transformer** com arquitetura encoder-decoder implementado do zero:

- **Positional Encoding** — codificação posicional sinusoidal
- **Encoder** — camadas com Multi-Head Attention + FFN + Layer Norm + Dropout
- **Decoder** — camadas com Self-Attention + Cross-Attention + FFN + Layer Norm
- **Learning Rate Schedule** — warmup linear seguido de decaimento inverso (baseado no paper *Attention Is All You Need*)

### Hiperparâmetros padrão

| Parâmetro | Valor |
|---|---|
| `num_layers` | 4 |
| `d_model` | 128 |
| `num_heads` | 4 |
| `dff` | 512 |
| `dropout` | 0.1 |
| `max_tokens` | 64 |

---

## 🚀 Executando as Rotinas (Quick Start)

As rotinas podem ser executadas individualmente com Docker Compose usando **profiles**.

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Preparar o Dataset

```bash
docker compose --profile prepare up --build
```

Baixa o dataset ParaCrawl EN-PT, tokeniza e gera TFRecords em `data/processed/`.

**Variáveis opcionais:**

| Variável | Padrão | Descrição |
|---|---|---|
| `DATASET_NAME` | `para_crawl/enpt` | Dataset do TFDS |
| `MAX_TOKENS` | `64` | Máximo de tokens por sentença |
| `TRAIN_RECORDS` | `5000` | Amostras de treino |
| `VAL_RECORDS` | `500` | Amostras de validação |
| `SEED` | `42` | Seed de reprodutibilidade |

### 2. Treinar o Modelo

```bash
docker compose --profile train up
```

Treina o Transformer e exporta o SavedModel versionado em `artifacts/<run_id>/`.

**Variáveis opcionais:**

| Variável | Padrão | Descrição |
|---|---|---|
| `EPOCHS` | `5` | Número de épocas |
| `BATCH_SIZE` | `32` | Tamanho do batch |
| `THRESHOLD` | `0.30` | Threshold de qualidade |
| `GIT_SHA` | `unknown` | SHA do commit para rastreabilidade |

### 3. Iniciar a API de Inferência

```bash
docker compose --profile api up
```

A API estará disponível em **http://localhost:8000**.

> **Dica:** Para especificar um modelo, defina `DEFAULT_RUN_ID=<run_id>`.

### 4. Executar os Testes

```bash
docker compose --profile tests up
```

---

## 📡 Endpoints da API

| Método | Rota | Descrição                                                          |
|---|---|--------------------------------------------------------------------|
| `GET` | `/health` | Health check — status, `run_id` e `model_loaded`                   |
| `GET` | `/model` | Retorna o `run_id` do modelo ativo                                 |
| `GET` | `/metrics` | Contadores: `requests_total`, `errors_total`, `translations_total` |
| `POST` | `/predict` | Traduz texto EN → PT                                               |
| `POST` | `/reload` | Hot-reload de modelo por `run_id` ou `artifacts_dir`               |
| `GET` | `/docs` | Documentação interativa (Swagger UI)                               |

### Exemplo — `/predict`

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, como vai você?"}'
```

**Response:**
```json
{
  "translation": "Hello, how are you?",
  "run_id": "nmt_20260225T130027Z_ap5vq0",
  "latency_ms": 42.5
}
```

---

## ⚙️ Variáveis de Ambiente (API)

| Variável | Padrão | Descrição |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts` | Diretório raiz dos artefatos |
| `DEFAULT_RUN_ID` | *(vazio)* | `run_id` do modelo a carregar na inicialização |
| `API_PORT` | `8000` | Porta exposta pelo Docker |

---

## 🧪 Testes

Os testes de contrato validam:

- ✅ `/health` retorna `200` com `status`, `run_id` e `model_loaded`
- ✅ `/predict` retorna `503`/`500` sem modelo carregado
- ✅ `/predict` valida texto vazio (`422`) e texto acima de 512 caracteres (`422`)
- ✅ `/metrics` retorna os três contadores como inteiros
- ✅ `/metrics` incrementa corretamente após chamadas de `/predict`
- ✅ `/model` retorna o campo `run_id`

---

## 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|---|---|
| **Python 3.11** | Linguagem principal |
| **TensorFlow 2.20** | Framework de ML (modelo + servindo) |
| **TensorFlow Text 2.20** | Tokenização |
| **TensorFlow Datasets** | Download do dataset ParaCrawl |
| **FastAPI** | Framework da API REST |
| **Uvicorn** | Servidor ASGI |
| **Pytest + HTTPX** | Testes automatizados |
| **Docker / Docker Compose** | Containerização e orquestração |

---

## 🗺️ Diagrama de Arquitetura

```mermaid
flowchart TD
    Client(["👤 Cliente"])
    GHA(["⚙️ GitHub Actions\nlint → test → build → publish"])
    GHCR(["📦 GHCR\nRegistry"])

    subgraph Docker Compose ["🐳 Docker Compose — rede mlops"]
        Nginx["🔀 NGINX\nGateway\n(Basic Auth + rate limit + logging)"]
        API["🤖 Inference API\nFastAPI :8000\n/predict /health /metrics /reload"]
        N8N["🔁 n8n\nOrquestrador :5678"]
        Prepare["📂 Preparar Dados\nTFRecords em data/processed"]
        Train["🧠 Treinar Modelo\nSavedModel + métricas + run_id"]
        Validate["✅ Validar\nthreshold de qualidade"]
        Publish["📦 Publicar Artefatos\npublished/<run_id> + registry/latest"]
        Deploy["🚀 Deploy\nPOST /reload com run_id"]
        Prometheus["📈 Prometheus\n:9090"]
        JsonExporter["🔄 json-exporter\n:7979"]
        Grafana["📊 Grafana\n:3000"]
        Loki["📋 Loki\n:3100"]
        Promtail["📨 Promtail\nColeta de logs dos containers"]
    end

    Client -->|"HTTP"| Nginx
    Nginx -->|"proxy"| API
    Client -->|"webhook POST"| N8N
    N8N -->|"1. docker run"| Prepare
    Prepare -->|"dados processados"| Train
    Train -->|"run_id + métricas"| Validate
    Validate -->|"aprovado"| Publish
    Publish -->|"artefato versionado"| Deploy
    Deploy -->|"POST /reload"| API
    JsonExporter -->|"scrape /metrics JSON"| API
    Prometheus -->|"scrape Prometheus format"| JsonExporter
    Grafana --> Prometheus
    Grafana --> Loki
    Grafana -->|"alert webhook"| N8N
    API -->|"stdout/stderr"| Promtail
    Nginx -->|"access/error logs"| Promtail
    Promtail -->|"push logs"| Loki
    GHA -->|"push image"| GHCR
```

---

## 🏗️ Infraestrutura Adicionada

| Componente | Tecnologia | Função |
|---|---|---|
| **Orquestrador** | n8n | Pipeline ML via webhook |
| **API Gateway** | NGINX | Basic Auth, rate limit e logging |
| **Métricas** | Prometheus + json-exporter + Grafana | Coleta, conversão de formato e dashboard |
| **Logs** | Promtail + Loki + Grafana | Centralização de logs |
| **CI/CD** | GitHub Actions | Lint, test, build, publish |

---

## 🧠 Decisões Técnicas

- **n8n como orquestrador central**: escolhido para tornar o pipeline visual, webhook-driven e fácil de demonstrar durante a avaliação.
- **NGINX como gateway**: solução simples e suficiente para atender autenticação, rate limiting e logging sem adicionar complexidade desnecessária.
- **Publicação de artefatos fora de `ml/`**: a etapa de publicação foi implementada em [`automation/`](automation/) para separar claramente automação de plataforma da camada base de ML.
- **Prometheus + json-exporter**: a API expõe métricas em JSON, então o `json-exporter` foi usado para integrar a coleta sem reimplementar os endpoints existentes.
- **Promtail + Loki**: adotados para centralização de logs de containers e visualização integrada no Grafana.
- **Docker socket no n8n**: opção prática para o contexto do desafio, permitindo que o workflow execute as etapas de forma reproduzível via containers.

---

## 🚀 Executando o Ambiente Completo

### Pré-requisitos

- Docker e Docker Compose
- Arquivo `.env` configurado a partir de `.env.example`
- Arquivo `nginx/.htpasswd` configurado a partir de `nginx/.htpasswd.example`

### 1. Subir o ambiente completo

```bash
docker compose --profile api up -d --build
```

Esse comando sobe a API e o gateway via profile `api`, além dos serviços de suporte sem profile, como `n8n`, `Prometheus`, `Grafana`, `Loki` e `Promtail`.

| Serviço | URL |
|---|---|
| API (via NGINX) | http://localhost |
| n8n | http://localhost:5678 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Loki readiness | http://localhost:3100/ready |

### 2. Disparar o pipeline via webhook

```bash
curl -X POST http://localhost:5678/webhook/mlops-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "para_crawl/enpt",
    "max_tokens": 64,
    "epochs": 10,
    "threshold": 0.30,
    "git_sha": "abc1234"
  }'
```

O pipeline executa automaticamente: **Preparar Dados → Treinar → Validar → Publicar → Deploy**.

Observação: as rotinas-base de ML foram preservadas; a etapa de publicação fica na camada de automação da plataforma.

### 3. Autenticação na API

A API está protegida por Basic Auth via NGINX. Use as credenciais configuradas no seu arquivo local `nginx/.htpasswd`:

```bash
curl -u admin:admin http://localhost/health
```

Para encerrar a stack do ambiente completo:

```bash
docker compose --profile api down --remove-orphans
```

---

## 📊 Observabilidade

- **Grafana** — acesse http://localhost:3000 (usuário: `admin`, senha: definida em `GF_ADMIN_PASSWORD`)
- Datasources provisionados automaticamente: Prometheus e Loki
- Métricas da API disponíveis em `/metrics` e coletadas pelo Prometheus a cada 15s
- Logs dos containers são coletados pelo Promtail e enviados ao Loki, incluindo a API e o gateway NGINX
- O gateway aplica `Basic Auth`, `limit_req` de `30r/m` por cliente e `access_log` em JSON
- O roteamento de alertas está provisionado para o webhook `mlops-alerts` no `n8n`

---

## 🔄 CI/CD

Pipeline GitHub Actions com 4 stages em sequência:

| Stage | Gatilho | Ação |
|---|---|---|
| **lint** | push/PR | `ruff check` em `ml/`, `inference_api/`, `tests/` |
| **test** | após lint | `pytest -q` |
| **build** | após test | `docker build` |
| **publish** | apenas `main` | push para GHCR |

---

## ⚖️ Trade-Offs e Limitações

- O foco desta entrega está em **MLOps, automação e operação da plataforma**, não em maximizar a qualidade do modelo base de tradução.
- As métricas `requests_total`, `errors_total` e `translations_total` são mantidas em memória pela API; ao recriar o container, esses contadores reiniciam.
- A publicação de artefatos foi implementada como um registro local versionado em disco, suficiente para o desafio, mas não substitui um model registry dedicado em produção.
- O uso de `docker.sock` no `n8n` é uma escolha pragmática para demonstração local; em um ambiente produtivo, eu substituiria isso por runners dedicados ou jobs isolados.
- Os alertas foram integrados ao Grafana e roteados ao webhook do `n8n`; para produção, eu complementaria com política de escalation, histórico e destinos externos adicionais.

---

## 🚀 Próximos Passos

- Persistir métricas da aplicação em backend próprio ou exportá-las nativamente em formato Prometheus
- Substituir a execução via `docker.sock` por workers/runners dedicados
- Adicionar rollback automatizado orientado por alerta ou health degradation
- Evoluir a publicação local de artefatos para um registry de modelos ou storage remoto versionado
- Expandir testes automatizados com cenários de integração do pipeline orquestrado

---

## 🔐 Variáveis de Ambiente

As configurações locais podem ser definidas em `.env`, com base em `.env.example`. Credenciais reais e arquivos locais sensíveis, como `nginx/.htpasswd`, não são versionados.
