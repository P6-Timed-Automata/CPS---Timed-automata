import collections


class State:
    """
    An instance of the class State is a state of an automaton.

    Attributes:
        name (str): name of the state
        initial (bool): true is the state is initial
        accepting (bool): true is the state is accepting
        edges_in (list[Edge]): set of the incoming edges
        edges_out (list[Edge]): set of the outgoing edges
    """
    def __init__(self, name: str, initial: bool = False, accepting: bool = False) -> None:
        """
        Args:
            name (str): Name of the state
            initial (:obj:`bool`, optional): True if the state is initial, False by default
            accepting (:obj:`bool`, optional): True if the state is accepting, False by default
        """
        self.name = name
        self.initial = initial
        self.accepting = accepting
        self.edges_in = list()
        self.edges_out = list()
        self.rank = 0

    def visit_number(self) -> int:
        """
        Returns:
            int: Number of time the state was visited
        """
        nb = 0
        for edge in self.edges_in:
            nb += edge.visit_number()
        return nb

    def add_edge(self, edge, in_out: str) -> None:
        """
        Add the edge to the set of incoming or outgoing edges of the state
        Args:
            edge (Edge): the edge to add
            in_out (str): "in" for incoming and "out" for outgoing
        """
        if in_out == 'in':
            self.edges_in.append(edge)
        elif in_out == 'out':
            self.edges_out.append(edge)

    def del_edge(self, edge, in_out: str) -> None:
        """
        Remove the edge of the set of incoming or outgoing edges of the state
        Args:
            edge (Edge): the edge to delete
            in_out (str): "in" for incoming and "out" for outgoing
        """
        if in_out == 'in':
            self.edges_in.remove(edge)
        elif in_out == 'out':
            self.edges_out.remove(edge)

    def search_edges(self, symbol: str, in_out: str) -> list:
        """
        Return the incoming or outgoing edges having a specified symbol
        Args:
            symbol (str): the researched symbol
            in_out (str): "in" for incoming and "out" for outgoing
        Returns:
            list[Union[Edge, None]]: List containing the edges if found
        """
        res = []
        if in_out == 'in':
            for edge in self.edges_in:
                if edge.symbol == symbol: res.append(edge)
        elif in_out == 'out':
            for edge in self.edges_out:
                if edge.symbol == symbol: res.append(edge)
        return res

    def undeterministic_edge_destination(self, timed: bool) -> list:
        """
        Check if the set of outgoing edges of the state is deterministic and returns a list of the destination states of the problematic edges
        Args:
            timed (bool): true if the two outgoing edges can have the same symbols but guards not overlapping, false if there can't be more that one transition for a given symbol
        Returns:
            list[State]: Destination states of the non determinist edges

        """
        symbols = [edge.symbol for edge in self.edges_out]
        res = dict(collections.Counter(symbols))
        for symb, occ in res.items():
            if occ > 1:
                if not timed:
                    return [edge.destination for edge in self.edges_out if edge.symbol == symb]
                rep = [e for e in self.edges_out if e.symbol == symb]
                for e in rep:
                    for other in [edge for edge in rep if edge != e]:
                        overlaps = (min(max(e.guard), max(other.guard)) - max(min(e.guard), min(other.guard)) >= 0)
                        if e.destination == other.destination: overlaps = True
                        if not e.destination.edges_out and not other.destination.edges_out: overlaps = True
                        if overlaps:
                            return [e.destination, other.destination]
        return list()
