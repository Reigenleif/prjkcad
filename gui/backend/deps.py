_harness = None


def get_harness():
    global _harness
    if _harness is None:
        try:
            from gui.backend.llm_harness import CADLLMHarness
        except ImportError:
            from llm_harness import CADLLMHarness
        _harness = CADLLMHarness()
    return _harness
