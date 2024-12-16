from typing import List
from .document_loading_base import DocumentLoadingBase
from ..helpers.azure_document_intelligence_helper import AzureDocumentIntelligenceClient
from ..common.source_document import SourceDocument
import urllib.parse


class Layout2DocumentLoading(DocumentLoadingBase):
    def __init__(self) -> None:
        super().__init__()

    def load(self, document_url: str) -> List[SourceDocument]:
        azure_document_intelligence_client = AzureDocumentIntelligenceClient()
        input_file_name = urllib.parse.unquote(document_url.split("?")[0].split("/")[-1])
        pages_content = azure_document_intelligence_client.analyze_layout(
            document_url,
            input_file_name,
            gen_image_description=True
        )
        documents = [
                        SourceDocument(
                            content=pages_content,
                            source=document_url,
                            offset=0,
                            page_number=1,
                        )
        ]
        # documents = [
        #     SourceDocument(
        #         content=page,
        #         source=document_url,
        #         offset=0,
        #         page_number=1,
        #     )
        #     for page in pages_content
        # ]
        return documents
