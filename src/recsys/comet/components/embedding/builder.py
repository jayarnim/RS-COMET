from .embedding import IDXEmbeddingWithHistory


def embedding_builder(**kwargs):
    cls = IDXEmbeddingWithHistory
    return cls(**kwargs)