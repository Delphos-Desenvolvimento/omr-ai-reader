# main.py

import cv2
from omr import OMRGrader, get_retangles

# -----------------------------------------------------------------------------
# PARÂMETROS DE CONFIGURAÇÃO GERAL
# -----------------------------------------------------------------------------
# Caminho para a imagem da PROVA COMPLETA
IMAGEM_PROVA_COMPLETA = "images_test\prova13.jpeg"

# Configurações do Corretor (OMR)
MODO_DEBUG = True
NUM_ALTERNATIVAS = 4

# --- MUDANÇA PRINCIPAL: LISTA DE GABARITOS ---
# Em vez de um único gabarito, criamos uma lista.
# Cada dicionário na lista corresponde a uma área (ROI) na ordem em que são detectadas (esquerda para a direita).
GABARITOS = [
    # Gabarito para a primeira área (ROI 1)
    {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 0, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2 },
    
    # Gabarito para a segunda área (ROI 2) - Exemplo com respostas diferentes
    {0: 2, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2},
    
    # Adicione mais gabaritos aqui se sua prova tiver mais de 2 áreas
    # { ... gabarito para ROI 3 ... }
]


# -----------------------------------------------------------------------------
# BLOCO DE EXECUÇÃO PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- ETAPA 1: Detectar e extrair as áreas de resposta (ROIs) ---
    print("[FLUXO] Iniciando ETAPA 1: Detecção das áreas de resposta...")
    rois_encontrados = get_retangles(IMAGEM_PROVA_COMPLETA, min_size=100)

    if not rois_encontrados:
        print("[ERRO] Nenhuma área de resposta foi encontrada. Encerrando o programa.")
    else:
        print(f"\n[FLUXO] ETAPA 1 Concluída: {len(rois_encontrados)} áreas encontradas.")
        
        # Verificação de segurança: garante que temos gabaritos suficientes
        if len(rois_encontrados) > len(GABARITOS):
            print(f"\n[ALERTA] Foram encontradas {len(rois_encontrados)} áreas, mas apenas {len(GABARITOS)} gabaritos foram fornecidos.")
            print("[INFO] A correção continuará apenas para as áreas que possuem um gabarito correspondente.")
            # Limita o número de ROIs a serem processados ao número de gabaritos disponíveis
            rois_encontrados = rois_encontrados[:len(GABARITOS)]

        print("\n[FLUXO] Iniciando ETAPA 2: Processamento de cada área...")
        print("-" * 60)

        # --- ETAPA 2 & 3: Iterar sobre cada ROI e processá-lo ---
        resultados_totais = []
        
        for i, roi_imagem in enumerate(rois_encontrados):
            print(f"\n=== PROCESSANDO ÁREA {i+1} ===")
            
            # Pega o gabarito específico para a área atual
            gabarito_atual = GABARITOS[i]
            print(f"[INFO] Usando Gabarito #{i+1}...")

            # Cria o corretor com as configurações
            grader = OMRGrader(
                answer_key=gabarito_atual,
                num_alternativas=NUM_ALTERNATIVAS,
                debug_mode=MODO_DEBUG
            )

            # Processa a área e obtém os resultados
            resultados = grader.processar_prova(roi_imagem, roi_index=i)
            
            if resultados:
                resultados_totais.append({
                    'area': i+1,
                    'resultados': resultados
                })
                
                # Exibe os resultados resumidos
                print(f"\n=== RESULTADOS ÁREA {i+1} ===")
                print(f"Retângulo detectado: {'Sim' if resultados['rectangle_detected'] else 'Não'}")
                print(f"Total de bolhas detectadas: {resultados['bubble_count']}")
                print(f"Questões corretas: {resultados['correct_answers']}/{resultados['total_questions']}")
                print(f"Pontuação: {resultados['score']:.2f}%")
                
                # Exibe as alternativas marcadas
                print("\nAlternativas marcadas:")
                for ans in resultados['marked_answers']:
                    status = "CORRETA" if ans['is_correct'] else f"INCORRETA (Resposta: {ans['correct']+1})"
                    print(f"  Questão {ans['question']}: Alternativa {ans['marked']+1} - {status}")
                
                print("-" * 60)

        # Fecha todas as janelas do OpenCV ao final do processamento
        if MODO_DEBUG:
            cv2.destroyAllWindows()
            
        print("\n[FLUXO] Processamento concluído. Resultados finais:")
        print("=" * 60)
        for resultado in resultados_totais:
            print(f"\nÁREA {resultado['area']}:")
            print(f"  - Retângulo detectado: {'Sim' if resultado['resultados']['rectangle_detected'] else 'Não'}")
            print(f"  - Bolhas detectadas: {resultado['resultados']['bubble_count']}")
            print(f"  - Acertos: {resultado['resultados']['correct_answers']}/{resultado['resultados']['total_questions']}")
            print(f"  - Pontuação: {resultado['resultados']['score']:.2f}%")
        
        print("\n[FLUXO] Programa finalizado.")


