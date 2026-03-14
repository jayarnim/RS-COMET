from .default import default_selector
from .freq import freq_selector
from .tfidf import tfidf_selector


SELECTOR_REGISTRY = {
    "default": default_selector,
    "freq": freq_selector,
    "tfidf": tfidf_selector,
}