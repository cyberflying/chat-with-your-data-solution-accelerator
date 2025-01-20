from typing import List
from .document_loading_base import DocumentLoadingBase
from ..helpers.azure_document_intelligence_helper import AzureDocumentIntelligenceClient
from ..common.source_document import SourceDocument
from urllib.parse import unquote, urlparse, parse_qs


class LayoutDocumentLoading(DocumentLoadingBase):
    def __init__(self) -> None:
        super().__init__()

    def load(self, document_url: str) -> List[SourceDocument]:
        azure_document_intelligence_client = AzureDocumentIntelligenceClient()
        input_file_name = unquote(document_url.split("?")[0].split("/")[-1])
        gen_figure_desc = parse_qs(urlparse(document_url).query).get("gen_figure_desc", ["false"])
        gen_figure_desc_bool = gen_figure_desc[0].lower() == "true"
        pages_content = azure_document_intelligence_client.analyze_layout(
            document_url,
            input_file_name,
            gen_figure_desc_bool
        )
        documents = [
            SourceDocument(
                content=pages_content,
                source=document_url,
                offset=0,
                page_number=1,
            )
        ]

        return documents
