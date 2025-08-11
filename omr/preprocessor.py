import cv2
import numpy as np

class DocumentProcessor:
    """
    Classe responsável por processar o documento (folha) em imagem.
    Combina a correção de perspectiva com as técnicas de binarização e
    morfologia do script de análise de gabarito.
    """
    def __init__(self, image_path, target_width=800):
        """
        Inicializa o processador de documento.

        Args:
            image_path (str): Caminho para o arquivo de imagem.
            target_width (int): Largura desejada para redimensionamento inicial.
        """
        self.image_path = image_path
        self.target_width = target_width
        self.original = None
        self.resized = None
        self.warped = None
        self.thresh = None
        self.processed_image = None  # Imagem final após todas as etapas

    def load_and_resize(self):
        """
        Carrega a imagem do caminho especificado e redimensiona
        mantendo a proporção com base em target_width.

        Returns:
            np.ndarray: Imagem redimensionada.
        """
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem em '{self.image_path}'. Verifique o caminho.")
        
        h, w = self.original.shape[:2]
        ratio = self.target_width / float(w)
        new_dim = (self.target_width, int(h * ratio))
        self.resized = cv2.resize(self.original, new_dim)
        return self.resized

    @staticmethod
    def order_points(pts):
        """
        Organiza quatro pontos em ordem: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts (np.ndarray): Array com 4 pontos (x, y).

        Returns:
            np.ndarray: Array reordenado de pontos.
        """
        rect = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # topo-esquerda
        rect[2] = pts[np.argmax(s)]      # baixo-direita
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # topo-direita
        rect[3] = pts[np.argmax(diff)]   # baixo-esquerda
        return rect

    def correct_perspective(self, min_area_ratio=0.5):
        """
        Detecta o maior contorno quadrilátero e aplica a transformação
        de perspectiva para alinhar a folha.

        Args:
            min_area_ratio (float): Razão mínima entre a área do contorno e a da imagem.

        Returns:
            np.ndarray: Imagem com perspectiva corrigida (deskewed).
        """
        gray = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        blur =cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        sheet = None
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) < min_area_ratio * (gray.shape[0] * gray.shape[1]):
                break
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                sheet = approx.reshape(4, 2)
                break

        if sheet is None:
            print("[AVISO] Nenhum contorno de folha detectado. Usando a imagem redimensionada.")
            self.warped = self.resized.copy()
            return self.warped

        rect = self.order_points(sheet)
        tl, tr, br, bl = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(rect, dst)
        self.warped = cv2.warpPerspective(self.resized, M, (maxW, maxH))
        return self.warped

    def apply_thresholding(self, blur_ksize=(5, 5), block_size=11, C=3):
        """
        Aplica desfoque e binarização adaptativa na imagem corrigida.
        Os parâmetros são baseados no script de correção de gabarito.

        Args:
            blur_ksize (tuple): Tamanho do kernel para o desfoque Gaussiano.
            block_size (int): Tamanho da vizinhança para cálculo do threshold.
            C (int): Constante subtraída da média.

        Returns:
            np.ndarray: Imagem binária (threshold).
        """
        if self.warped is None:
            raise ValueError("A correção de perspectiva deve ser executada primeiro.")
            
        gray = cv2.cvtColor(self.warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
        
        self.thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C
        )
        return self.thresh

    def apply_morphological_closing(self, kernel_size=9):
        """
        Aplica a operação de fechamento morfológico para unir bolhas próximas,
        exatamente como no script de correção de gabarito.

        Args:
            kernel_size (int): Diâmetro do elemento estruturante elíptico.

        Returns:
            np.ndarray: Imagem processada final.
        """
        if self.thresh is None:
            raise ValueError("A binarização (thresholding) deve ser executada primeiro.")
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.processed_image = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)
        return self.processed_image


if __name__ == "__main__":
    # --- ETAPA 1: Configure o caminho para sua imagem aqui ---
    IMAGE_PATH = "prova9.jpeg"  

    try:
        # --- ETAPA 2: Cria o objeto e executa o pipeline de pré-processamento ---
        print(f"[INFO] Processando a imagem: {IMAGE_PATH}")
        processor = DocumentProcessor(image_path=IMAGE_PATH, target_width=800)

        # 2.1 - Carrega e redimensiona a imagem
        resized_img = processor.load_and_resize()
        
        # 2.2 - Corrige a perspectiva da folha
        warped_img = processor.correct_perspective()
        
        # 2.3 - Aplica binarização com os parâmetros do primeiro script
        thresh_img = processor.apply_thresholding(blur_ksize=(5, 5), block_size=7, C=3)
        
        # 2.4 - Aplica fechamento morfológico para unir bolhas
        final_img = processor.apply_morphological_closing(kernel_size=3)

        print("[INFO] Pré-processamento concluído. Exibindo resultados...")
        
        # --- ETAPA 3: Exibe os resultados de cada etapa ---
        cv2.imshow("1 - Imagem Redimensionada", resized_img)
        cv2.imshow("2 - Perspectiva Corrigida", warped_img)
        cv2.imshow("3 - Binarizacao (Threshold)", thresh_img)
        cv2.imshow("4 - Imagem Final Processada (com Fechamento)", final_img)

        # Mantém as janelas abertas até que uma tecla seja pressionada
        print("[INFO] Pressione qualquer tecla para fechar todas as janelas.")
        cv2.waitKey(0)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        cv2.destroyAllWindows()


