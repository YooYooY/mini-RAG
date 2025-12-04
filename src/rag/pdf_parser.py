from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from paddleocr import PaddleOCR

from src.core.config import (
    OCR_LANG,
    OCR_USE_ANGLE,
)


class PDFParser:
    """
    自动解析 PDF（文字版 + 扫描版）
    文字版：PyPDFLoader
    扫描版：PaddleOCR（自动 fallback）
    """

    def __init__(self):
        # M1/M2/M3 自动开启 accelerated
        # lang='ch' 用于解析中文 PDF
        self.ocr = PaddleOCR(use_angle_cls=OCR_USE_ANGLE, lang=OCR_LANG)

    def parse(self, file_path: Path) -> List[Document]:
        """
        统一解析 API：
        - 文本 PDF → PyPDFLoader
        - 扫描 PDF → OCR fallback
        返回 Document 列表
        """
        docs = self._try_load_text_pdf(file_path)

        if docs:
            return docs

        # fallback to OCR
        return self._ocr_pdf(file_path)

    # ----------------------
    # 内部方法：PyPDFLoader
    # ----------------------
    def _try_load_text_pdf(self, file_path: Path) -> List[Document]:
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

            # docs 可能不是空，但 page_content 全是空白 → 识别为扫描 PDF
            if any(d.page_content.strip() for d in docs):
                print("使用 PyPDFLoader（文字版 PDF）")
                return docs

            print("PyPDFLoader 解析失败（可能是扫描 PDF）")
            return []
        except:
            return []

    # ----------------------
    # 内部方法：OCR 模式
    # ----------------------
    def _ocr_pdf(self, file_path: Path) -> List[Document]:
        print("使用 PaddleOCR 解析扫描 PDF...")

        results = self.ocr.ocr(str(file_path))

        extracted_text = ""
        for page in results:
            for line in page:
                text = line[1][0]  # paddles 的文字内容
                extracted_text += text + "\n"

        if not extracted_text.strip():
            raise ValueError("OCR 解析失败：PDF 内无可识别文字")

        return [Document(page_content=extracted_text)]
