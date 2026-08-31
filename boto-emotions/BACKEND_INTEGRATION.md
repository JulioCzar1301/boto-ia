# Integracao Backend - BotoLens Emotion API

Este documento define o contrato esperado para integrar a deteccao de emocoes ao backend do BotoLens.

## Objetivo

Expor a IA de emocoes em Python como servico HTTP, mantendo imagens apenas em memoria durante a inferencia. O backend nao deve salvar imagens, payloads base64, videos ou frames.

## Fluxo recomendado

```text
Android
  POST /emotion/session/start
  abre camera
  envia JPEG/base64 a 3-5 FPS para /emotion/session/{session_id}/frame
  exibe EmotionResponse
  DELETE /emotion/session/{session_id} ao encerrar
```

## Endpoints obrigatorios

- `GET /health`: status operacional.
- `GET /privacy`: contrato tecnico de privacidade e retencao.
- `POST /emotion/session/start`: cria sessao temporal.
- `POST /emotion/session/{session_id}/frame`: processa frame da sessao.
- `DELETE /emotion/session/{session_id}`: encerra sessao.

## Politica tecnica de privacidade

O servico deve manter:

- estado temporal numerico;
- scores suavizados;
- historico curto de features;
- buffer temporal da BiLSTM;
- contagem e timestamps da sessao.

O servico deve descartar apos a inferencia:

- imagem recebida;
- payload base64;
- frame OpenCV decodificado.

O servico nao deve:

- logar request body;
- salvar base64;
- salvar frames;
- escrever imagens em disco;
- usar `saida_teste` no fluxo de API;
- criar dataset a partir das requisicoes do app.

## Variaveis de ambiente

Variavel | Padrao | Uso
--- | --- | ---
`CORS_ORIGINS` | `*` | Origens permitidas.
`MAX_IMAGE_BASE64_CHARS` | `2500000` | Limite maximo do payload de imagem em base64.
`SESSION_TTL_SECONDS` | `300` | Tempo de vida de sessoes inativas.

## Payload recomendado

```json
{
  "image": "BASE64_JPEG",
  "include_debug": false,
  "use_temporal_as_final": false
}
```

Para reduzir custo:

- enviar JPEG redimensionado;
- testar inicialmente 8-10 FPS para preservar melhor as nuances temporais;
- reduzir para 3-5 FPS apenas se rede/backend pesarem demais;
- manter a mesma sessao durante a tela de camera;
- deletar a sessao ao sair.

## Teste manual pelo computador

A pasta `test_client/` contem uma pagina HTML para abrir a camera do proprio PC e enviar frames para a API, simulando o comportamento do app.

1. Suba a API no cluster ou localmente.
2. No PC de teste, sirva a pagina em `localhost`:

```bash
cd test_client
python -m http.server 8088
```

3. Abra:

```text
http://localhost:8088/webcam_emotion_test.html
```

4. Preencha a URL base da API:

```text
http://IP_DO_SERVIDOR:8003
```

5. Abra a camera, crie a sessao e inicie o envio.

Observacao: navegadores permitem camera em `localhost`. Se a pagina estiver em HTTPS, a API tambem deve usar HTTPS para evitar bloqueio por mixed content.
