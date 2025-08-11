# circle.py (MODIFICADO)

from imutils import contours
import numpy as np
import imutils
import cv2

class OMRGrader:
    """
    Classe para corrigir provas de múltipla escolha.
    AGORA ACEITA CAMINHO DE IMAGEM OU OBJETO DE IMAGEM EM MEMÓRIA.
    """
    def __init__(self, answer_key, num_alternativas=4, debug_mode=False,
                 min_bubble_width=25, min_bubble_height=25,
                 min_bubble_ratio=0.8, max_bubble_ratio=1.5,
                 merge_kernel_size=9, proximity_dist=20):
        self.answer_key = answer_key
        self.num_alternativas = num_alternativas
        self.debug_mode = debug_mode
        self.min_bubble_width = min_bubble_width
        self.min_bubble_height = min_bubble_height
        self.min_bubble_ratio = min_bubble_ratio
        self.max_bubble_ratio = max_bubble_ratio
        self.merge_kernel_size = merge_kernel_size
        self.proximity_dist = proximity_dist
        self.image = None
        self.paper = None
        self.gray = None
        self.thresh_closed = None
        self.question_contours = []
        self.median_radius = 20

    def _carregar_e_preprocessar(self, imagem_entrada):
        """Carrega a imagem de um caminho OU objeto e aplica pré-processamento."""
        print("[INFO] Carregando e pré-processando a área de resposta...")

        if isinstance(imagem_entrada, str): # Se a entrada for um texto (caminho)
            self.image = cv2.imread(imagem_entrada)
            if self.image is None:
                print(f"[ERRO] Não foi possível carregar a imagem do caminho: {imagem_entrada}")
                return False
        else: # Senão, assume que é um objeto de imagem (numpy array)
            self.image = imagem_entrada

        self.image = cv2.resize(self.image, (400, 800))
        self.paper = self.image.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            11, 5
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.merge_kernel_size, self.merge_kernel_size)
        )
        self.thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        if self.debug_mode:
            cv2.imshow("Threshold Fechado", self.thresh_closed)

        return True

    def _detectar_e_agrupar_bolhas(self):
        """Encontra, filtra e agrupa os contornos que representam as bolhas."""
        print("[INFO] Procurando e agrupando contornos das bolhas...")
        cnts = cv2.findContours(
            self.thresh_closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        rawCnts = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if (w >= self.min_bubble_width and h >= self.min_bubble_height and
                ar >= self.min_bubble_ratio and ar <= self.max_bubble_ratio):
                rawCnts.append((c, (x, y, w, h)))

        merged = []
        for c, (x, y, w, h) in rawCnts:
            cx, cy = x + w/2, y + h/2
            found = False
            for idx, (mx, my, mw, mh, members) in enumerate(merged):
                dist = np.hypot(cx - mx, cy - my)
                if dist < self.proximity_dist:
                    nx, ny = min(x, mx - mw/2), min(y, my - mh/2)
                    nx2, ny2 = max(x + w, mx + mw/2), max(y + h, my + mh/2)
                    nmw, nmh = nx2 - nx, ny2 - ny
                    nmx, nmy = nx + nmw/2, ny + nmh/2
                    merged[idx] = (nmx, nmy, nmw, nmh, members + [c])
                    found = True
                    break
            if not found:
                merged.append((cx, cy, w, h, [c]))

        final_contours = [cv2.convexHull(np.vstack(members)) for _, _, _, _, members in merged]
        self.question_contours = final_contours
        print(f"[INFO] {len(self.question_contours)} bolhas únicas encontradas.")

        radii = [max(cv2.boundingRect(c)[2:]) / 2 for c in self.question_contours]
        if radii:
            self.median_radius = int(np.median(radii))
        
        if self.debug_mode:
            self._mostrar_bolhas_detectadas()

    def _mostrar_bolhas_detectadas(self):
        """Mostra uma janela de depuração com as bolhas detectadas."""
        print("[INFO] Modo de depuração: mostrando bolhas detectadas...")
        debug_image = self.image.copy()
        cv2.drawContours(debug_image, self.question_contours, -1, (255, 0, 0), 2)
        cv2.imshow("Bolhas Detectadas (Debug)", debug_image)
        cv2.waitKey(0)

    def _ordenar_e_corrigir(self):
        """Ordena as bolhas por questão e corrige a prova."""
        results = {
            'rectangle_detected': len(self.question_contours) > 0,
            'bubble_count': len(self.question_contours),
            'marked_answers': [],
            'correct_answers': 0,
            'total_questions': len(self.answer_key)
        }
        
        if len(self.question_contours) < len(self.answer_key) * self.num_alternativas:
            print(f"[AVISO] Número de bolhas ({len(self.question_contours)}) é menor que o esperado.")
        
        self.question_contours = contours.sort_contours(self.question_contours, method="top-to-bottom")[0]

        for (q, i) in enumerate(np.arange(0, len(self.question_contours), self.num_alternativas)):
            if q >= results['total_questions']:
                break
            
            cnts_q = contours.sort_contours(self.question_contours[i:i + self.num_alternativas])[0]
            
            # Lista para armazenar a contagem de pixels preenchidos para cada bolha
            bubble_scores = []
            bubble_areas = []
            
            # Calcular o preenchimento para cada bolha
            for (j, c) in enumerate(cnts_q):
                # Calcular a área da bolha
                area = cv2.contourArea(cv2.convexHull(c))
                bubble_areas.append(area)
                
                # Calcular o preenchimento
                mask = np.zeros(self.thresh_closed.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                filled = cv2.countNonZero(cv2.bitwise_and(self.thresh_closed, self.thresh_closed, mask=mask))
                fill_ratio = filled / area if area > 0 else 0
                bubble_scores.append((fill_ratio, j, filled, area))
            
            # Ordenar por pontuação (maior primeiro)
            bubble_scores.sort(reverse=True, key=lambda x: x[0])
            
            # Calcular a diferença percentual entre a primeira e segunda bolha mais escuras
            marked_answer_idx = -1
            if len(bubble_scores) > 1 and bubble_scores[0][0] > 0:
                # Calcular a diferença percentual em relação à segunda maior pontuação
                diff_ratio = (bubble_scores[0][0] - bubble_scores[1][0]) / bubble_scores[0][0]
                
                # Definição dos limiares
                MIN_FILL_THRESHOLD = 0.4  # Mínimo de preenchimento para considerar uma bolha marcada
                MIN_DIFF_RATIO = 0.15   # Diferença mínima para considerar uma bolha como marcada
                
                # Debug: mostrar informações das bolhas
                if self.debug_mode:
                    print(f"\n=== Questão {q+1} ===")
                    for idx, (ratio, j, filled, area) in enumerate(bubble_scores):
                        print(f"Bolha {j+1}: {filled} pixels preenchidos de {area:.0f} ({ratio*100:.1f}%)")
                    print(f"Diferença entre 1ª e 2ª: {diff_ratio*100:.1f}%")
                    print(f"Limite mínimo de diferença: {MIN_DIFF_RATIO*100}%")
                    print(f"Limite mínimo de preenchimento: {MIN_FILL_THRESHOLD*100}%")
                print(f"MIN_FILL_THRESHOLD {diff_ratio}")
                print(f"MIN_DIFF_RATIO {bubble_scores[0][0]}")
                # Verificar se a diferença é significativa e se o preenchimento é suficiente
                if diff_ratio > MIN_DIFF_RATIO and bubble_scores[0][0] > MIN_FILL_THRESHOLD:
                   
                    marked_answer_idx = bubble_scores[0][1]
                    if self.debug_mode:
                        print(f"Bolha {marked_answer_idx+1} marcada com confiança!")
                elif self.debug_mode:
                    if diff_ratio <= MIN_DIFF_RATIO:
                        print("Nenhuma bolha marcada: diferença insuficiente entre as duas primeiras bolhas")
                    if bubble_scores[0][0] <= MIN_FILL_THRESHOLD:
                        print(f"Nenhuma bolha marcada: preenchimento máximo ({bubble_scores[0][0]*100:.1f}%) abaixo do limite mínimo")
            
            correct_answer_idx = self.answer_key[q]
            
            results['marked_answers'].append({
                'question': q + 1,
                'marked': marked_answer_idx,
                'correct': correct_answer_idx,
                'is_correct': marked_answer_idx == correct_answer_idx
            })
            
            if marked_answer_idx == correct_answer_idx:
                results['correct_answers'] += 1
                
            if self.debug_mode and marked_answer_idx != -1:
                
                color = (0, 255, 0) if marked_answer_idx == correct_answer_idx else (0, 0, 255)
                cnt = cnts_q[correct_answer_idx]
                M = cv2.moments(cnt)
                center = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00'] != 0 else tuple(map(int, cv2.minEnclosingCircle(cnt)[0]))
                cv2.circle(self.paper, center, self.median_radius, color, 2)
        
        results['score'] = (results['correct_answers'] / results['total_questions']) * 100 if results['total_questions'] > 0 else 0
        return results

    def processar_prova(self, imagem_entrada, roi_index=None):
        """Executa todo o fluxo de correção da prova."""
        if not self._carregar_e_preprocessar(imagem_entrada):
            return None

        self._detectar_e_agrupar_bolhas()
        results = self._ordenar_e_corrigir()
        
        if self.debug_mode:
            
            self._mostrar_resultado_debug(results, roi_index)
            
        return results
        
    def _mostrar_resultado_debug(self, results, roi_index=None):
        """Mostra os resultados em modo debug."""
        titulo_janela = "Prova Corrigida"
        if roi_index is not None:
            titulo_janela = f"ROI {roi_index+1} Corrigida"
            print(f"\n--- Resultado ROI {roi_index+1} ---")
        else:
            print("\n--------------------------")
        
        print(f"[INFO] Retângulo detectado: {'Sim' if results['rectangle_detected'] else 'Não'}")
        print(f"[INFO] Total de bolhas detectadas: {results['bubble_count']}")
        print(f"[INFO] Questões corretas: {results['correct_answers']}/{results['total_questions']}")
        print(f"[INFO] Pontuação: {results['score']:.2f}%")
        print("[INFO] Alternativas marcadas:")
        for ans in results['marked_answers']:
            status = "CORRETA" if ans['is_correct'] else f"INCORRETA (Resposta: {ans['correct']+1})"
            print(f"  Questão {ans['question']}: Alternativa {ans['marked']+1} - {status}")
        
        if self.debug_mode:
            cv2.putText(self.paper, f"{results['score']:.2f}%", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Original", self.image)
            cv2.imshow(titulo_janela, self.paper)
            cv2.waitKey(0)
            pass
        
    
if __name__ == "__main__":
    # --- DEFINA OS PARÂMETROS GERAIS AQUI ---
    CAMINHO_DA_IMAGEM = "roi_color_1.png"
    MODO_DEBUG = True
    ANSWER_KEY = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 0, 7: 2, 8: 2, 9: 2}
    NUM_ALTERNATIVAS = 4

    # --- Cria uma instância do corretor de provas ---
    grader = OMRGrader(
        answer_key=ANSWER_KEY,
        num_alternativas=NUM_ALTERNATIVAS,
        debug_mode=MODO_DEBUG
        # Outros parâmetros podem ser passados aqui se necessário,
        # ex: merge_kernel_size=11
    )

    # --- Processa a imagem da prova ---
    grader.processar_prova(CAMINHO_DA_IMAGEM)


