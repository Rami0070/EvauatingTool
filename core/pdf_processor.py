import io
import pymupdf as fitz
from PIL import Image
from typing import List, Dict, Any, Tuple

def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    """
    PDF 바이트 데이터를 받아 각 페이지를 고해상도 PIL Image 리스트로 변환합니다.
    DPI 150~200 수준으로 손글씨 및 시각 자료(그래프, 그림 등) 판독성을 확보합니다.
    """
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img.convert("RGB"))
    finally:
        doc.close()
    return images

def convert_multiple_pdfs_to_page_items(pdf_files: List[Any], dpi: int = 150) -> List[Dict[str, Any]]:
    """
    여러 업로드된 PDF 파일들로부터 페이지별 이미지와 원본 정보를 추출합니다.
    반환 구조:
    [
        {
            "filename": 파일명,
            "page_num": 페이지 번호(1부터 시작),
            "image": PIL Image 객체
        },
        ...
    ]
    """
    all_pages = []
    for uploaded_file in pdf_files:
        filename = getattr(uploaded_file, "name", "document.pdf")
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            file_bytes = uploaded_file.read()
        else:
            file_bytes = uploaded_file
            
        imgs = convert_pdf_to_images(file_bytes, dpi=dpi)
        for idx, img in enumerate(imgs):
            all_pages.append({
                "filename": filename,
                "page_num": idx + 1,
                "image": img
            })
    return all_pages
