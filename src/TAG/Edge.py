from src.TAG.State import State


class Edge:
    """
    An instance of the class Edge is an edge of an automaton.

    Attributes:
        source (State): source state of the edge
        destination (State) : destination state of the edge
        symbol (str) : label of the edge
        guard (list) list of accepted time value for the edge
        tss (Dict[int, list(tuple)])
        proba (float): probability of
    """
    def __init__(self, source: State, destination: State, symbol: str, guard: list) -> None:
        """
        Args:
            source (State): source State of the edge
            destination (State): destination State of the edge
            symbol (str): label of the edge
            guard (list): list of accepted time value of the edge
        """
        self.source = source
        self.destination = destination
        self.symbol = symbol
        self.guard = guard
        self.tss = dict()
        self.proba = 0
        source.add_edge(self, 'out')
        destination.add_edge(self, 'in')

    def visit_number(self) -> int:
        """
        Returns:
            int: Number of time this transition has been visited
        """
        nb = 0
        for key, value in self.tss.items():
            nb += len(value)
        return nb

    def reduced_guard(self) -> list:
        """
        Return the interval for the delay
        Returns:
            tuple[int, int]: the minimal and maximal values for the delay
        """
        mini = min(self.guard)
        maxi = max(self.guard)
        return [mini, maxi]

    def print(self, reduce_guard: bool = True) -> str:
        """
        Print the edge in the following format
        "SOURCE_STATE -> DESTINATION_STATE [label='SYMBOL GUARD']"
        Args:
            reduce_guard (:obj:`bool`, optional): true for a reduced printing of the guard (min and max), false for all the possible time values
        """
        tmp = self.source.name + ' -> ' + self.destination.name + ' [label="' + self.symbol + ' '
        if reduce_guard:
            tmp += str(self.reduced_guard()) + '"]'
        else:
            tmp += str(self.guard) + '"]'
        print(tmp)
        return tmp

    def reduce_gtime(self) -> tuple:
        """
        Return the interval for the global clock
        Returns:
            tuple[int, int]: the minimal and maximal values for the global clock
        """
        l = [t[1] for v in self.tss.values() for t in v]
        if len(l) == 0: return ("", "")
        return (min(l), max(l))
