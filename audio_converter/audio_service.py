import os
import tempfile
import json
import re
from typing import Optional, Tuple, Dict, Any

from google import genai


ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg"}


def _allowed_audio(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text or "", re.IGNORECASE)
    candidate = fenced.group(1) if fenced else (text or "")
    candidate = candidate.replace("```", "").strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        inner = candidate[start : end + 1]
        try:
            return json.loads(inner)
        except Exception:
            return None
    return None


def analyze_audio_request(file_storage, reference_text: Optional[str]) -> Tuple[Dict[str, Any], int]:
    if not file_storage:
        return {"status": "no_file", "message": "Nenhum arquivo de áudio enviado"}, 200

    if file_storage.filename == "":
        return {"status": "no_file", "message": "Nenhum arquivo de áudio selecionado"}, 200

    if not _allowed_audio(file_storage.filename):
        return {
            "status": "invalid_file_type",
            "message": f"Tipo de arquivo não permitido. Use: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        }, 200

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") 
    if not api_key:
        return {
            "status": "config_error",
            "message": "GOOGLE_API_KEY não configurada no ambiente."
        }, 500

    try:
        client = genai.Client(api_key=api_key)

        suffix = os.path.splitext(file_storage.filename)[1] or ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_storage.read())
            temp_audio_path = tmp.name

        try:
            uploaded = client.files.upload(file=temp_audio_path)

            base_instructions = (
                "Você é um avaliador de leitura em voz alta. Analise a fluência, pronúncia, entonação, ritmo, pausas, "
                "clareza e erros. Se um texto de referência for fornecido, realize o alinhamento e compare o que foi "
                "lido com o texto (palavras faltantes, inseridas, trocadas), tempo total e velocidade (palavras por minuto). "
                "Responda ESTRITAMENTE em JSON no formato especificado a seguir; cada avaliação vai de 0 a 10."
            )

            schema_example = {
                "overall_score": 0.0,
                "fluency": {"score": 0.0, "feedback": ""},
                "pronunciation": {"score": 0.0, "feedback": ""},
                "intonation": {"score": 0.0, "feedback": ""},
                "rhythm": {"score": 0.0, "feedback": ""},
                "pauses": {"score": 0.0, "feedback": ""},
                "clarity": {"score": 0.0, "feedback": ""},
                "duration_seconds": 0.0,
                "words_per_minute": 0.0,
                "reference_used": bool(bool(reference_text)),
                "alignment": {
                    "words_total": 0,
                    "words_read": 0,
                    "words_missing": [],
                    "words_inserted": [],
                    "words_substituted": [],
                    "accuracy_percent": 0.0
                },
                "suggestions": []
            }

            contents = [base_instructions, f"Formato JSON obrigatório: {json.dumps(schema_example)}"]
            if reference_text:
                contents.append(f"Texto de referência (PT-BR):\n{reference_text}")
            contents.append(uploaded)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )

            parsed = _extract_json_from_text(response.text or "")
            evaluation = parsed if parsed is not None else {"raw": (response.text or "").replace("```", "").strip()}

            return {"status": "success", "evaluation": evaluation}, 200
        finally:
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass
    except Exception as e:
        return {"status": "processing_error", "message": f"Erro ao analisar áudio: {str(e)}"}, 500
