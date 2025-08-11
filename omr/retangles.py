from .preprocessor import DocumentProcessor
import cv2

class RectangleDetector:
    def __init__(self, thresh_img, min_size=150, max_size=800):
        self.thresh = thresh_img
        self.min_size = min_size
        self.max_size = max_size
        self.rects = []
        self.grouped = []

    def detect(self):
        contours, _ = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if self.min_size < w < self.max_size and self.min_size < h < self.max_size:
                    self.rects.append([x, y, w, h])
        return self.rects

    def group(self, group_threshold=1, eps=0.2):
        if self.rects:
            self.grouped, _ = cv2.groupRectangles(self.rects, group_threshold, eps)
            
            # --- LINHA ADICIONADA ---
            # Ordena os retângulos agrupados pela coordenada X (da esquerda para a direita)
            self.grouped = sorted(list(self.grouped), key=lambda r: r[0])
            
        return self.grouped

    def get_rois(self, source_img=None, as_thresh=True):
        rois = []
        img = self.thresh if as_thresh or source_img is None else source_img
        for x, y, w, h in self.grouped:
            roi = img[y:y+h, x:x+w]
            rois.append(roi)
        return rois

    def draw(self, image):
        img_copy = image.copy()
        for i, (x, y, w, h) in enumerate(self.grouped):
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Adiciona o número da ordem para visualização
            cv2.putText(img_copy, f"ROI {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img_copy

if __name__ == '__main__':
    # Altere o nome da imagem conforme necessário
    IMAGE_PATH = "images_test\prova10.jpeg"

    try:
        # --- ETAPA 1: Processar o documento ---
        print(f"[INFO] Processando a imagem: {IMAGE_PATH}")
        processor = DocumentProcessor(image_path=IMAGE_PATH)
        processor.load_and_resize()
        processor.correct_perspective()
      
        processor.apply_thresholding(blur_ksize=(3, 3), block_size=5, C=3)
        processor.apply_morphological_closing(kernel_size=4)
        
        # --- ETAPA 2: Detectar os retângulos na imagem processada ---
        print("[INFO] Detectando retângulos...")
        # Passa a imagem final do processador para o detector
        detector = RectangleDetector(processor.processed_image, min_size=100) # Ajuste min_size se necessário
        detector.detect()
        detector.group()
        print(f"[INFO] {len(detector.grouped)} retângulos encontrados.")

        # --- ETAPA 3: Visualizar os resultados ---
        # Desenha os retângulos na imagem com a perspectiva corrigida
        output_image = detector.draw(processor.warped)
        
        cv2.imshow("Imagem Processada (Entrada para Detector)", processor.processed_image)
        cv2.imshow("Retangulos Detectados", output_image)
        color_rois = detector.get_rois(source_img=processor.warped, as_thresh=False)

        # Salva cada um
        for idx, roi in enumerate(color_rois[:2]):      # garante só 2
            filename = f'roi_color_{idx+1}.png'
            ok = cv2.imwrite(filename, roi)
            if ok:
                print(f'[INFO] Salvou recorte colorido {idx+1} em: {filename}')
            else:
                print(f'[ERRO] Falha ao salvar {filename}')

        
        # Opcional: Extrair e mostrar cada ROI (Região de Interesse)
        thresh_rois = detector.get_rois()
        for idx, roi in enumerate(thresh_rois):
            cv2.imshow(f'ROI {idx}', roi)
            filename = f'roi_{idx+1}.png'         
            cv2.imwrite(filename, roi) 
        
        cv2.waitKey(0)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        cv2.destroyAllWindows()


