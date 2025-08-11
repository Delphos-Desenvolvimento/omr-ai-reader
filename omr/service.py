import os
import tempfile
import json
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from . import get_retangles, OMRGrader, transformar_gabaritos

# Configurações da API
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Certifique-se de que a pasta de upload existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_omr_image(image_path: str, NUM_ALTERNATIVAS: int = 4, GABARITOS: Optional[list] = None) -> Dict[str, Any]:
    """
    Processa a imagem OMR e retorna os resultados.
    Apenas retorna resultados se exatamente 2 retângulos forem detectados
    e todas as bolhas esperadas forem encontradas em cada retângulo.
    """
    try:
        # Tenta ler a imagem
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "no_image", "message": "Não foi possível ler a imagem"}
        except Exception as e:
            return {"status": "invalid_image", "message": f"Erro ao processar a imagem: {str(e)}"}

        # Tenta detectar áreas de resposta
        try:
            rois_encontrados = get_retangles(image_path, min_size=100)
        except Exception as e:
            return {"status": "detection_error", "message": f"Erro na detecção de áreas: {str(e)}"}
        
        # Verifica se exatamente 2 retângulos foram encontrados
        if not rois_encontrados or len(rois_encontrados) != 2:
            return {
                "status": "invalid_rectangles", 
                "message": f"Número incorreto de retângulos detectados. Esperado: 2, Encontrado: {len(rois_encontrados) if rois_encontrados else 0}"
            }
        
        resultados = []
        todas_bolhas_ok = True
        
        # Processa cada área encontrada
        for i, roi_imagem in enumerate(rois_encontrados):
            if i >= len(GABARITOS or []):
                break
                
            gabarito_atual = GABARITOS[i]
            total_bolhas_esperado = len(gabarito_atual) * NUM_ALTERNATIVAS
            
            try:
                grader = OMRGrader(
                    answer_key=gabarito_atual,
                    num_alternativas=NUM_ALTERNATIVAS,
                    debug_mode=False
                )
                
                # Processa a área
                resultado = grader.processar_prova(roi_imagem, roi_index=i)
                
                if resultado:
                    # Verifica se detectou todas as bolhas esperadas
                    bolhas_detectadas = resultado["bubble_count"]
                    todas_bolhas_detectadas = (bolhas_detectadas == total_bolhas_esperado)
                    
                    # Se alguma área não tiver todas as bolhas, marca como não OK
                    if not todas_bolhas_detectadas:
                        todas_bolhas_ok = False
                    
                    # Formata o resultado
                    resultado_formatado = {
                        "area": i + 1,
                        "status": "complete" if todas_bolhas_detectadas else "incomplete",
                        "retangulo_detected": True,
                        "bolhas_detectadas": bolhas_detectadas,
                        "bolhas_esperadas": total_bolhas_esperado,
                        "todas_bolhas_detectadas": todas_bolhas_detectadas,
                        "respostas": []
                    }
                    
                    # Se detectou todas as bolhas, inclui as respostas
                    if todas_bolhas_detectadas and "marked_answers" in resultado:
                        respostas = []
                        tem_resposta_baixa_confianca = False
                        
                        for ans in resultado["marked_answers"]:
                            # Verifica se a resposta tem baixa confiança (marked = -1)
                            if ans["marked"] == -1:
                                tem_resposta_baixa_confianca = True
                            
                            respostas.append({
                                "questao": ans["question"],
                                "alternativa_marcada": ans["marked"] + 1 if ans["marked"] != -1 else None,
                                "alternativa_correta": ans["correct"] + 1,
                                "correto": ans["is_correct"]
                            })
                        
                        resultado_formatado["respostas"] = respostas
                        
                        # Se alguma resposta tiver baixa confiança, marca como não OK
                        if tem_resposta_baixa_confianca:
                            todas_bolhas_ok = False
                    
                    resultados.append(resultado_formatado)
                
            except Exception:
                # Em caso de erro no processamento de uma área, marca como não OK
                todas_bolhas_ok = False
                resultados.append({
                    "area": i + 1,
                    "status": "error",
                    "message": f"Erro no processamento da área {i+1}",
                    "retangulo_detected": True,
                    "bolhas_detectadas": 0,
                    "bolhas_esperadas": total_bolhas_esperado,
                    "todas_bolhas_detectadas": False,
                    "respostas": []
                })
        
        # Verifica se todas as áreas foram processadas com sucesso e todas as bolhas foram detectadas
        if not resultados or not todas_bolhas_ok:
            # Verifica se o motivo foi baixa confiança nas respostas
            baixa_confianca = any(
                any(resp.get("alternativa_marcada") is None for resp in res.get("respostas", []))
                for res in resultados
            )
            
            return {
                "status": "incomplete_detection",
                "message": "garanta que as bolhas foram bem preenchidas" if baixa_confianca 
                          else "Não foram detectadas todas as bolhas necessárias",
                "resultados": resultados
            }
            
        return {
            "status": "success",
            "message": "Processamento concluído com sucesso",
            "resultados": resultados
        }
        
    except Exception as e:
        # Captura qualquer outro erro inesperado
        return {
            "status": "unexpected_error",
            "message": f"Erro inesperado: {str(e)}",
            "resultados": []
        }


def process_request(file_storage, gabarito_json_str: Optional[str]) -> Tuple[Dict[str, Any], int]:
    if not file_storage:
        return {"status": "no_file", "message": "Nenhum arquivo enviado"}, 200

    if file_storage.filename == '':
        return {"status": "no_file", "message": "Nenhum arquivo selecionado"}, 200

    if not gabarito_json_str:
        return {"status": "bad_request", "message": "O campo 'gabarito' é obrigatório no formulário."}, 400

    try:
        gabarito_recebido = json.loads(gabarito_json_str)
    except json.JSONDecodeError:
        return {"status": "bad_request", "message": "O gabarito fornecido não é um JSON válido."}, 400
    except Exception as e:
        return {"status": "processing_error", "message": f"Erro ao processar o gabarito: {str(e)}"}, 500

    GABARITOS = transformar_gabaritos(gabarito_recebido)

    if file_storage and allowed_file(file_storage.filename):
        try:
            # Read image file directly from memory
            file_data = file_storage.read()
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_filename = temp_file.name
                cv2.imwrite(temp_filename, img)

            try:
                # Process the OMR image
                resultado = process_omr_image(temp_filename, NUM_ALTERNATIVAS=4, GABARITOS=GABARITOS)
                return resultado, 200
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_filename)
                except Exception:
                    pass
        except Exception as e:
            return {"status": "processing_error", "message": f"Erro ao processar a imagem: {str(e)}"}, 200

    return {"status": "invalid_file_type", "message": f"Tipo de arquivo não permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"}, 200


