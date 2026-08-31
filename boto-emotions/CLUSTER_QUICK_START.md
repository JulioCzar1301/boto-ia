# BotoLens Emotion Server - Cluster Quick Start

Esta pasta contem apenas o necessario para subir a API de deteccao de emocoes no servidor/cluster.

## Porta

```text
8002 -> BotoLens Emotion API
```

## Subir servidor

```bash
docker-compose up --build -d
```

O Dockerfile instala PyTorch CPU-only para evitar baixar pacotes CUDA grandes durante o build.

## Verificar status

```bash
curl http://localhost:8002/health
curl http://localhost:8002/privacy
```

Se estiver acessando de outra maquina:

```bash
curl http://IP_DO_SERVIDOR:8002/health
curl http://IP_DO_SERVIDOR:8002/privacy
```

## Endpoints principais

```text
GET    /health
GET    /privacy
POST   /emotion/session/start
POST   /emotion/session/{session_id}/frame
DELETE /emotion/session/{session_id}
```

## Contrato de privacidade

- A API recebe a imagem somente para inferencia em memoria.
- A API nao salva imagem, base64, video ou frame.
- A API nao deve logar corpo das requisicoes.
- A sessao guarda apenas estado numerico temporario.
- Sessoes inativas expiram por `SESSION_TTL_SECONDS`.

## Configuracoes

No `docker-compose.yml`:

```text
CORS_ORIGINS=*
MAX_IMAGE_BASE64_CHARS=2500000
SESSION_TTL_SECONDS=300
```

## Teste com cliente externo

Do computador que tiver camera, use a URL base:

```text
http://IP_DO_SERVIDOR:8002
```

O cliente de teste HTML nao precisa ficar no servidor. Ele pode rodar no PC local.
