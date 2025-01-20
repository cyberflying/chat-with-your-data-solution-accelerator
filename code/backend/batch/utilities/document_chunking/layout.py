from typing import List
from .document_chunking_base import DocumentChunkingBase
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .chunking_strategy import ChunkingSettings
from ..common.source_document import SourceDocument


class LayoutDocumentChunking(DocumentChunkingBase):
    def __init__(self) -> None:
        pass

    def chunk(
        self, documents: List[SourceDocument], chunking: ChunkingSettings
    ) -> List[SourceDocument]:
        full_document_content = "".join(
            list(map(lambda document: document.content, documents))
        )
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunked_content_list = text_splitter.split_text(full_document_content)

        # Split chunks recursively if they are too large
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap
        )
        refined_chunks = []
        for chunk in chunked_content_list:
            if len(chunk.page_content) > 8000:
                refined_chunks.extend(recursive_splitter.split_text(chunk.page_content))
            else:
                refined_chunks.append(chunk.page_content)

        document_url = documents[0].source
        # Create document for each chunk
        documents = []
        chunk_offset = 0
        for idx, chunked_content in enumerate(refined_chunks):
            documents.append(
                SourceDocument.from_metadata(
                    content=chunked_content,
                    document_url=document_url,
                    metadata={"offset": chunk_offset},
                    idx=idx,
                )
            )

            chunk_offset += len(chunked_content)
        return documents
