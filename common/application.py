class Application:

    def __init__(self):  # type:ignore[reportMissingSuperCall]
        pass

    def fitness(self, processor, network) -> float:
        raise NotImplementedError

    def validation(self, processor, network):
        return None
