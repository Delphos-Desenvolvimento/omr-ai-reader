## Colins IA — API de OMR e Análise de Leitura (Áudio)

API em Flask com documentação Swagger para:

- Correção automática de gabaritos por OMR (detecção de bolhas em imagens)
- Análise de leitura oral a partir de áudio, com avaliação de fluência, pronúncia e alinhamento opcional com texto de referência (via Google Gemini)

Abra a documentação interativa em: `http://localhost:5000/apidocs`


### Sumário

- Visão geral
- Requisitos
- Instalação e execução
- Variáveis de ambiente
- Endpoints da API (com exemplos)
- Estrutura do projeto
- Dicas de uso e troubleshooting


### Visão geral

- O serviço OMR recebe uma imagem da folha de respostas e um gabarito em JSON, detecta duas áreas (ROIs) e identifica bolhas marcadas, comparando com as respostas corretas.
- O serviço de áudio recebe um arquivo (`mp3`, `wav`, `m4a`, `ogg`) e, opcionalmente, um texto de referência para alinhar o que foi lido. Retorna um JSON com pontuação e feedback por critério.


### Requisitos

- Python 3.10+ (recomendado)
- Pip e virtualenv
- Dependências listadas em `requirements.txt`
- Para o endpoint de áudio: uma chave de API do Google Gemini válida


### Instalação e execução

Windows (cmd.exe):

```bat
cd "C:\Users\devBr\OneDrive\Área de Trabalho\colins ia"
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

REM Configurar a chave (veja seção Variáveis de ambiente)
set GOOGLE_API_KEY=SEU_TOKEN

python app.py
```

Acesse:

- API: `http://localhost:5000/`
- Swagger UI: `http://localhost:5000/apidocs`


### Variáveis de ambiente

- `GOOGLE_API_KEY` ou `GEMINI_API_KEY`: chave do Google Gemini para uso no endpoint de áudio.

Notas:

- Não utilize chaves hardcoded em produção. Sempre configure via variável de ambiente.
- O código tenta ler `GOOGLE_API_KEY` ou `GEMINI_API_KEY`.


### Endpoints da API

Todos os endpoints estão documentados no Swagger:

- OMR: `omr/docs/omr_process.yml`
- Áudio: `omr/docs/audio_analyze.yml`
- Health: `omr/docs/health.yml`


#### 1) Processar OMR

- Método: POST
- Rota: `/api/processar-omr`
- Consome: `multipart/form-data`
- Campos do formulário:
  - `file` (file) — imagem da prova (`png`, `jpg`, `jpeg`)
  - `gabarito` (string ou arquivo JSON) — lista de gabaritos, um por ROI detectada

Formato do `gabarito` (exemplo):

```json
[
  {"1": "a", "2": "b", "3": "b", "4": "b", "5": "c"},
  {"1": "c", "2": "c", "3": "b"}
]
```

Observações:

- Cada objeto no array corresponde a uma área (ROI) na ordem de detecção (esquerda para direita).
- Letras válidas: `a`, `b`, `c`, `d`. O serviço converte letras para números internamente via `omr/utils.py::transformar_gabaritos`.

Exemplos de chamada (Windows):

- Recomendado: enviar o gabarito como arquivo para evitar problemas de aspas no Windows cmd.

```bat
REM Salve o JSON acima em gabarito.json
curl.exe -X POST http://localhost:5000/api/processar-omr ^
  -F "file=@C:\caminho\para\prova.jpg" ^
  -F "gabarito=@gabarito.json;type=application/json"
```

Resposta (exemplo resumido):

```json
{
  "status": "success",
  "message": "Processamento concluído com sucesso",
  "resultados": [
    {
      "area": 1,
      "status": "complete",
      "retangulo_detected": true,
      "bolhas_detectadas": 40,
      "bolhas_esperadas": 40,
      "todas_bolhas_detectadas": true,
      "respostas": [
        { "questao": 1, "alternativa_marcada": 2, "alternativa_correta": 2, "correto": true }
      ]
    }
  ]
}
```

Possíveis status:

- `success`: processamento concluído
- `incomplete_detection`: nem todas as bolhas foram detectadas ou há respostas com baixa confiança (sem marcação clara)
- `invalid_rectangles`: não foram detectados exatamente 2 retângulos
- `bad_request`: gabarito inválido (JSON malformado, letras fora de `a` a `d`, chaves não numéricas, etc.)


#### 2) Analisar Áudio

- Método: POST
- Rota: `/api/analisar-audio`
- Consome: `multipart/form-data`
- Campos do formulário:
  - `audio` (file) — formatos aceitos: `mp3`, `wav`, `m4a`, `ogg`
  - `texto` (string) — opcional, texto de referência para alinhamento/comparação

Exemplo de chamada (Windows):

```bat
curl.exe -X POST http://localhost:5000/api/analisar-audio ^
  -F "audio=@C:\caminho\para\leitura.m4a" ^
  -F "texto=Hoje é um bom dia para estudar."
```

Resposta (exemplo resumido):

```json
{
  "status": "success",
  "evaluation": {
    "overall_score": 6.8,
    "fluency": { "score": 6.0, "feedback": "..." },
    "pronunciation": { "score": 7.0, "feedback": "..." },
    "intonation": { "score": 6.0, "feedback": "..." },
    "rhythm": { "score": 6.5, "feedback": "..." },
    "pauses": { "score": 6.0, "feedback": "..." },
    "clarity": { "score": 7.0, "feedback": "..." },
    "duration_seconds": 8.4,
    "words_per_minute": 136.0,
    "reference_used": true,
    "alignment": {
      "words_total": 19,
      "words_read": 19,
      "words_missing": [],
      "words_inserted": [],
      "words_substituted": [],
      "accuracy_percent": 100.0
    },
    "suggestions": ["..."]
  }
}
```

Erros comuns:

- `no_file`: sem arquivo enviado
- `invalid_file_type`: extensão não permitida
- `config_error`: variável `GOOGLE_API_KEY` não configurada
- `processing_error`: falha interna durante a análise


#### 3) Healthcheck

- Método: GET
- Rota: `/`
- Retorna uma string e está documentado em `omr/docs/health.yml`


### Estrutura do projeto

```
colins ia/
  app.py                      # Inicializa Flask e define rotas; integra Swagger
  requirements.txt            # Dependências
  audio_converter/
    audio_service.py          # Lógica de análise de áudio com Google Gemini
  omr/
    service.py                # Fluxo principal do OMR (leitura de imagem, validações, retorno)
    utils.py                  # Conversão de gabaritos de letras -> números
    preprocessor.py           # Classe utilitária para pré-processamento de imagens (deskew, threshold, morfologia)
    docs/
      omr_process.yml         # Especificação Swagger do endpoint OMR
      audio_analyze.yml       # Especificação Swagger do endpoint de áudio
      health.yml              # Especificação Swagger do healthcheck
  scripts/
    main.py                   # Exemplo de uso local do OMR em imagem com debug
  uploads/                    # Pasta padrão para uploads (garantida no código)
```


### Dicas de uso e troubleshooting

- Use imagens nítidas, com preenchimento consistente das bolhas. Detecções incompletas normalmente ocorrem por marcações muito leves ou ruído.
- O endpoint OMR espera detectar exatamente 2 retângulos (duas áreas) na folha. Se a sua folha tiver outro layout, ajuste a lógica em `omr/service.py`.
- Para Windows PowerShell, use `curl.exe` para evitar conflito com o alias `Invoke-WebRequest`.
- Para enviar JSON em `multipart/form-data` no Windows, prefira o upload via arquivo (`@gabarito.json;type=application/json`).
- Configuração da chave do Gemini: exporte `GOOGLE_API_KEY`/`GEMINI_API_KEY` antes de subir a API.


### Desenvolvimento

- Rode `python app.py` para um servidor de desenvolvimento (porta 5000). O `debug=True` já está habilitado no código.
- Para testar o pipeline de OMR localmente sem API, ajuste `IMAGEM_PROVA_COMPLETA` e `GABARITOS` em `scripts/main.py` e execute o script.




