from torch import nn


def _convert_str2class(m_str: str, modules: dict):
    if m_str is None:
        return None
    
    if hasattr(nn, m_str):
        m = getattr(nn, m_str)  # from torch modules
    elif m_str in modules:
        m = modules[m_str]  # from local modules
    else:
        raise AssertionError(f"could not find any matching module for '{m_str}'")
    
    return m