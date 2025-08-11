from .preprocessor import DocumentProcessor
from .retangles import RectangleDetector
import cv2

def get_retangles(IMAGE_PATH, min_size=100):
    """
    Detecta retângulos na imagem fornecida e retorna os recortes (ROIs) em memória.

    Args:
        IMAGE_PATH (str): Caminho para a imagem de entrada.
        min_size (int): Tamanho mínimo do retângulo.

    Returns:
        list of numpy.ndarray: Lista de imagens ROI (coloridas) dos retângulos detectados.
    """
    print(f"[INFO] Processando a imagem: {IMAGE_PATH}")
    # Processamento do documento
    processor = DocumentProcessor(image_path=IMAGE_PATH)
    processor.load_and_resize()
    processor.correct_perspective()
    processor.apply_thresholding(blur_ksize=(3, 3), block_size=5, C=3)
    processor.apply_morphological_closing(kernel_size=4)

    # Detecção de retângulos
    detector = RectangleDetector(processor.processed_image, min_size=min_size)
    detector.detect()
    detector.group()
    print(f"[INFO] {len(detector.grouped)} retângulos encontrados.")

    # Extrai os ROIs coloridos a partir da imagem corrigida
    color_rois = detector.get_rois(source_img=processor.warped, as_thresh=False)

    # Retorna a lista de ROIs em memória
    return color_rois

if __name__ == "__main__":
    rois = get_retangles("prova5.jpeg", min_size=100)
    for idx, roi in enumerate(rois):
        # por exemplo, exibir com OpenCV:
        cv2.imshow(f"ROI {idx+1}", roi)
    cv2.waitKey(0)


