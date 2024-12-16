from .chunking_strategy import ChunkingStrategy
from .layout import LayoutDocumentChunking
from .layout2 import Layout2DocumentChunking
from .page import PageDocumentChunking
from .fixed_size_overlap import FixedSizeOverlapDocumentChunking
from .paragraph import ParagraphDocumentChunking


def get_document_chunker(chunking_strategy: str):
    if chunking_strategy == ChunkingStrategy.LAYOUT.value:
        return LayoutDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.LAYOUT2.value:
        return Layout2DocumentChunking()
    elif chunking_strategy == ChunkingStrategy.PAGE.value:
        return PageDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.FIXED_SIZE_OVERLAP.value:
        return FixedSizeOverlapDocumentChunking()
    elif chunking_strategy == ChunkingStrategy.PARAGRAPH.value:
        return ParagraphDocumentChunking()
    else:
        raise Exception(f"Unknown chunking strategy: {chunking_strategy}")
