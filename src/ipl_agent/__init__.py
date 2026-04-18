__all__ = ["IPLAgent"]


def __getattr__(name: str):
    if name == "IPLAgent":
        from .agent import IPLAgent

        return IPLAgent
    raise AttributeError(name)
