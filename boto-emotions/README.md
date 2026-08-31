# BotoLens Emotion API

Versao experimental do modulo de deteccao de emocoes preparada para backend FastAPI. Esta pasta foi criada como copia separada do pipeline modular original para testar integracao via servidor sem alterar a base estavel.

## Visao geral

A API carrega os modelos uma unica vez e expoe endpoints HTTP para processar frames enviados em base64. Para fluxos de video, a API suporta sessoes persistentes, mantendo historico temporal, suavizacao, tracking de rosto e buffer da BiLSTM entre frames.

## Componentes principais

```text
FastAPI App (porta 8002)
  |
  | base64 frame
  v
Emotion Service
  |
  | ENet + MediaPipe + Fusion + Temporal State + BiLSTM
  v
EmotionResponse JSON
```

## Estrutura

```text
app/
  main.py              # Endpoints FastAPI
  schemas.py           # Schemas Pydantic
  services/
    emotion.py         # Singleton dos modelos e gerenciamento de sessoes
  utils/
    image.py           # Conversao base64 -> frame OpenCV

processor.py           # Pipeline original, agora com render_overlay opcional
models.py              # Carregamento ENet e MediaPipe
temporal_model.py      # BiLSTM temporal
Dockerfile
docker-compose.yml
```

## Como executar localmente

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Health check:

```bash
curl http://localhost:8002/health
```

Contrato de privacidade/retencao:

```bash
curl http://localhost:8002/privacy
```

## Como executar com Docker

```bash
docker-compose up --build
```

O build instala PyTorch/Torchvision CPU-only para evitar downloads de dependencias CUDA grandes.

A API ficara disponivel em:

```text
http://localhost:8002
```

## Endpoints

### `GET /health`

Retorna status da API, dispositivo usado e quantidade de sessoes ativas.

```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": true,
  "active_sessions": 0,
  "privacy_mode": "in_memory_inference_only"
}
```

### `GET /privacy`

Retorna o contrato tecnico de privacidade do servico. A API aceita imagens apenas para inferencia em memoria, nao salva frames, nao loga payload base64 e mantem somente estado numerico temporario da sessao.

### `POST /emotion/frame`

Processa um frame avulso ou um frame dentro de uma sessao existente.

Request:

```json
{
  "image": "BASE64_DA_IMAGEM",
  "session_id": "crianca-001",
  "include_debug": false,
  "use_temporal_as_final": false
}
```

Response:

```json
{
  "detected": true,
  "emotion": "happy",
  "confidence": 0.82,
  "scores": {
    "happy": 0.82,
    "sadness": 0.04,
    "anger": 0.03,
    "surprise": 0.06,
    "neutral": 0.05
  },
  "rule_based_emotion": "happy",
  "rule_based_confidence": 0.82,
  "temporal": {
    "ready": true,
    "emotion": "happy",
    "confidence": 0.78
  },
  "quality": {
    "quality_warning": false,
    "quality_penalty_active": false
  },
  "bbox": [120, 80, 420, 410],
  "face_count": 1,
  "session_id": "crianca-001",
  "debug": null
}
```

### `POST /emotion/session/start`

Cria uma sessao persistente.

```json
{
  "session_id": "crianca-001",
  "use_temporal_as_final": false
}
```

### `POST /emotion/session/{session_id}/frame`

Processa frame usando obrigatoriamente a sessao informada na URL. Esse e o endpoint recomendado para video continuo.

### `POST /emotion/session/{session_id}/reset`

Reinicia o estado temporal da sessao.

### `GET /emotion/session/{session_id}`

Consulta metadados da sessao.

### `DELETE /emotion/session/{session_id}`

Remove uma sessao.

## Padrao de uso recomendado no app

1. Criar sessao quando a camera iniciar.
2. Enviar frames compactados em JPEG/base64 inicialmente a 8-10 FPS.
3. Usar `/emotion/session/{session_id}/frame` durante a captura.
4. Remover ou resetar a sessao ao sair da tela.

## Teste com camera do PC

Para testar sem Android, use o cliente HTML em `test_client/webcam_emotion_test.html`.

```bash
cd test_client
python -m http.server 8088
```

Depois abra:

```text
http://localhost:8088/webcam_emotion_test.html
```

Na tela, informe a URL base da API. Exemplos:

```text
http://localhost:8002
http://IP_DO_CLUSTER:8002
```

## Configuracoes

Variavel | Padrao | Descricao
--- | --- | ---
`CORS_ORIGINS` | `*` | Origens permitidas, separadas por virgula.
`MAX_IMAGE_BASE64_CHARS` | `2500000` | Limite maximo do payload de imagem em base64.
`SESSION_TTL_SECONDS` | `300` | Tempo de vida de sessoes inativas.

## Observacoes

- O backend roda sem desenhar overlay para reduzir custo por frame.
- A imagem recebida e usada apenas em memoria durante a inferencia e descartada depois do processamento.
- A API nao salva imagem, base64, video ou frame em disco, banco, cache ou log.
- As sessoes mantem somente estado numerico temporario e expiram por inatividade.
- O pipeline visual por webcam continua disponivel em `main.py`.
- A pasta original `botolens_modularizado` nao e alterada por este experimento.

Para detalhes de integracao com o backend, veja `BACKEND_INTEGRATION.md`.
