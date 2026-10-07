from typing import Optional
from gui.backend.llm_harness import DualSeqLLMHarness

_harness: Optional[DualSeqLLMHarness] = None


def get_harness() -> DualSeqLLMHarness:
    global _harness
    if _harness is None:
        _harness = DualSeqLLMHarness(lazy_load=True)
    return _harness
