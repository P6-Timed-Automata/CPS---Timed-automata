from TAG.Automaton import Automaton
from TAG.State import State
from TAG.Edge import Edge
from typing import Union
import re
from datetime import datetime
from random import shuffle


class TALearner:
    def __init__(self, tss_path: str = None, tss_list: list = None, k: int = 2, res_path: str = None, display: bool = False, debug: bool = False, splits: bool = True, order='breadth-first') -> None:
        """
        Args:
            tss_path (str): path of the file containing the timed strings
            tss_list (list[str]): python list containing the timed strings
            k (:obj:`int`, optional): number of transitions to consider for the merging process, 2 by default
            res_path (:obj:`str`, optional): path to the file where to export the learned automaton
            display (:obj:`bool`, optional): true if the learned automaton should be displayed at the end, False by default
            debug (:obj:`bool`, optional): true if verbose mode needed, False by default
            splits (:obj:`bool`, optional): True if splits are allowed, True by default
            order (:obj:`str`, optional): Ordering method for the operations (breadth-first/depth-first/random/bottom-up), breadth-first by default
        """
        # t0 = datetime.now()
        self.tss = self.__import_tss(tss_path, tss_list)
        self.k = k
        self.debug = debug
        self.ta = Automaton()
        self.ta.tss = self.tss
        self.mode = order
        self.operations = {"merges": 0, "splits": 0}
        self.__phase_i()
        self.ta.show(title='After phase i')
        if splits:
            self.__phase_ii()
            self.ta.show(title='After phase ii')
        if self.ta.inconsistency_nb(self.tss, True) > 0:
            print("Automaton not consistent.")
        else:
            if res_path is not None:
                self.ta.export_ta(res_path)
            if display: self.ta.show(title='Final automaton')
        # dur = datetime.now() - t0
        # print('Execution time: ' + str(dur))

    def __reorder_states(self) -> None:
        """
        Reorder the states in function to the ordering method choosen
        """
        if self.mode == 'breadth-first':
            self.ta.states = sorted(self.ta.states, key=lambda s: s.rank)
        elif self.mode == 'bottom-up':
            self.ta.states = sorted(self.ta.states, key=lambda s: s.rank, reverse=True)
        elif self.mode == 'random':
            shuffle(self.ta.states)
        else:  # depth-first
            self.ta.states = sorted(self.ta.states, key=lambda s: eval(s.name[1:]))

    def __build_apta(self) -> None:
        """
        Initialize the automaton with one possible path per state \n
        """
        state_index = 1
        for i in range(0, len(self.tss)):
            last = 'S0'
            gtime = 0
            # self.ta.save_img(filename="img" + str(self.cpt), bstate=[s for s in self.ta.states if s.initial]); self.cpt += 1
            for index, pair in enumerate(self.tss[i]):
                pair = pair.split(':')  # pair[0]: symbol; pair[1]: time_value
                gtime += eval(pair[1])
                if pair[0] not in self.ta.symbols: self.ta.symbols.append(pair[0])
                res = self.ta.next_edge(last, pair[0])
                if res is None:
                    next = 'S' + str(state_index)
                    edge = self.ta.add_edge(last, next, pair[0], [eval(pair[1])])
                    edge.tss[i] = [(index, gtime)]
                    state_index += 1
                    self.ta.search_state(next).rank = index
                else:
                    next = res.destination.name
                    res.guard.append(eval(pair[1]))
                    if i in res.tss.keys():
                        res.tss[i].append((index, gtime))
                    else:
                        res.tss[i] = [(index, gtime)]
                if index == len(self.tss[i]) - 1: self.ta.search_state(next).accepting = True
                last = next
                # self.ta.save_img(filename="img" + str(self.cpt), bstate=[next], bedge=[res]); self.cpt += 1

   
        self.ta.update_probas()

    def __compute_future(self, states: list, symbols: list, futures: dict, level: int = 1) -> dict:
        """
        Compute the k-future of each state of the automaton \n
        Args:
            states (list[State]): list of maximal length k+1 containing a state sequence possible starting for the first state
            symbols (list[str]): list of maximal length k containing a symbol sequence possible starting from the first state
            futures (dict[State:list[tuple[list[State], list[str]]]]): dictionary containing the k-future of the states
            level (int): level of recursion
        Returns:
            dict: Dictionary containing the k-future of the states of the automaton
        """
        if level <= self.k and len(states[-1].edges_out) > 0:
            for edge in states[-1].edges_out:
                if level == 1:
                    symbols = []
                    states = states[:1]
                else:
                    symbols = symbols[:level - 1]
                    states = states[:level]
                states.append(edge.destination)
                symbols.append(edge.symbol)
                futures = self.__compute_future(states, symbols, futures, level + 1)
        else:
            if level == 1: return futures  # TODO: ARG
            if futures[states[0]] is None: futures[states[0]] = list()
            futures[states[0]].append((states, symbols))
        return futures

    @staticmethod
    def __compare_futures(future_1: list, future_2: list) -> bool:
        """
        Compare the k-futures of two states \n
        Args:
            future_1 (list[tuple(list[State], list[str])]): a k-future
            future_2 (list[tuple(list[State], list[str])]): another k-future
        Returns:
            bool: True if the k-futures are identicals
        """
        seq_symbol_1 = set([tuple(f[1]) for f in future_1])
        seq_symbol_2 = set([tuple(f[1]) for f in future_2])
        return bool(seq_symbol_1 == seq_symbol_2)

    def __edges_merge(self, edge_1: Edge, edge_2: Edge) -> None:
        """
        Merge two edges
        Args:
            edge_1 (Edge): First edge to merge
            edge_2 (Edge): Second edge to merge
        """
        if edge_2 not in self.ta.edges or edge_1 not in self.ta.edges:
            return
        if self.debug:
            print('Merging following edges:')
            edge_1.print()
            edge_2.print()
        # self.ta.save_img(filename="img" + str(self.cpt), bedge=[edge_1, edge_2], color="firebrick"); self.cpt += 1
        edge_1.guard += edge_2.guard
        for ts in edge_2.tss.keys():
            if ts in edge_1.tss.keys():
                edge_1.tss[ts] += edge_2.tss[ts]
            else:
                edge_1.tss[ts] = edge_2.tss[ts]
        self.ta.edges.remove(edge_2)
        edge_2.destination.del_edge(edge_2, 'in')
        edge_2.source.del_edge(edge_2, 'out')
        self.ta.update_probas()
        # self.ta.save_img(filename="img" + str(self.cpt), bedge=[edge_1], color="firebrick"); self.cpt += 1

    def __determinize_prefix(self, edge: Edge, timed: bool) -> None:  # after each merge
        """
        If merging two states induced a determinism issue in the prefix, solves it
        Args:
            edge (Edge): An incomming edge of a merged state
            timed (bool): True if time values should be taken into consideration
        """
        while True:
            edges = edge.source.search_edges(edge.symbol, 'out').copy()
            if timed:
                for e in edge.source.search_edges(edge.symbol, 'out'):
                    overlaps = False
                    for other in [e for e in edges if e != e]:
                        if self.__overlaps(e.guard, other.guard): overlaps = True
                    if overlaps: edges.remove(e)
            if len(edges) < 2: return
            state_1, state_2 = edges[0].destination, edges[1].destination
            if state_1 == state_2: self.__edges_merging(state_1, timed)
            else: self.__state_merging(state_1, state_2)
            if edge not in self.ta.edges: return

    @staticmethod
    def __overlaps(list_1: list, list_2: list) -> bool:
        """
        Tests if two lists overlap
        Args:
            list_1 (list[int]): First list
            list_2 (list[int]): Second list
        Returns:
            bool: True if the two lists overlap
        """
        return (min(max(list_1), max(list_2)) - max(min(list_1), min(list_2)) >= 0)

    def __edges_merging(self, state: State, timed) -> None:
        """
        Merge the edges having "state" as destination and with the same source and the same symbol\n
        Args:
            state (State): The destination state of the edges to merge
        """
        for symbol in self.ta.symbols:
            edges = state.search_edges(symbol, 'in')
            if len(edges) > 1:
                edges_per_state = dict()
                for edge in edges: edges_per_state.setdefault(edge.source, []).append(edge)
                edges_per_state = {k: v for k, v in edges_per_state.items() if len(v) > 1}
                for source, edges in edges_per_state.items():
                    while len(edges) > 1:
                        self.__edges_merge(edges[0], edges[1])
                        self.__determinize_prefix(edges[0], timed)
                        edges.remove(edges[1])

    def __merge(self, state_1:State, state_2:State, timed:bool=True) -> None:
        """
        Merge two states and determinization process
        Args:
            state_1 (State): First state to merge
            state_2 (State): Second state to merge
            timed (:obj:`bool`, optional): True if time should be taken into consideration, True by default

        Returns:

        """
        state = self.__state_merging(state_1, state_2, timed)
        res = state.undeterministic_edge_destination(False)
        deterministic = False if len(res) > 0 else True
        while not deterministic:
            if res[0] == res[1]:
                self.__edges_merging(res[0], timed)
            else:
                self.__merge(res[0], res[1], timed)
            if state not in self.ta.states: return  # state re-merged during the determinization process
            res = state.undeterministic_edge_destination(False)
            deterministic = False if len(res) > 0 else True

    def __state_merging(self, state_1: State, state_2: State, timed:bool=True) -> Union[State, None]:
        """
        Merges two states
        Args:
            state_1 (State): First state to merge
            state_2 (State): Second state to merge
            timed (:obj:`bool`, optional): True if time should be taken into consideration, True by default
        Returns:
            Union[State, None]: The state resultant of the merge, None if no merge
        """
        if state_1 not in self.ta.states or state_2 not in self.ta.states:
            return
        if eval(state_1.name[1:]) < eval(state_2.name[1:]):
            old_state, new_state = state_2, state_1
        else:
            old_state, new_state = state_1, state_2
        # self.ta.save_img(filename="img" + str(self.cpt), bstate=[old_state, new_state], color="firebrick"); self.cpt += 1
        old_state.rank = new_state.rank = min(old_state.rank, new_state.rank)
        for edge in old_state.edges_in:
            edge.destination = new_state
            new_state.add_edge(edge, 'in')
        for edge in old_state.edges_out:
            edge.source = new_state
            new_state.add_edge(edge, 'out')
        self.ta.states.remove(old_state)
        if old_state.accepting: new_state.accepting = True
        self.ta.update_probas()
        self.operations["merges"] += 1
        # self.ta.save_img(filename="img" + str(self.cpt), bstate=[new_state], color="firebrick"); self.cpt += 1
        return new_state

    def __solve_undeterminism(self, timed: bool) -> None:
        """
        Solves deterministic issues
        Args:
            timed (bool): True if time should be taken into consideration
        """
        complete = False
        while not complete:
            complete = True
            for state in self.ta.states:
                if len(state.undeterministic_edge_destination(timed)) > 0:  # Merge of 2 states with incomming edge of same symbol & from same source
                    res = state.undeterministic_edge_destination(timed)
                    if res[0] == res[1]:
                        self.__edges_merging(res[0], timed)
                        complete = False
                    else:
                        self.__merge(res[0], res[1], timed)
                        complete = False

    @staticmethod
    def __min_max(liste: list) -> list:
        """
        Return the minimum and the maximum of a list
        Args:
            liste (list[int]): Liste
        Returns:
            list[int]: minimum and the maximum of the list
        """
        return [min(liste), max(liste)]

    def __look_for_merges(self, timed: bool) -> bool:  # TODO: vérifier fonction
        """
        Looks for merges to perform
        Args:
            timed (bool): True is time should be taken into consideration
        Returns:
            bool: True if a merge was performed, False otherwise
        """
        self.__reorder_states()
        futures = dict.fromkeys(self.ta.states)
        for state in self.ta.states:
            states = [state]
            symbols = []
            futures = self.__compute_future(states, symbols, futures)
        for state_1 in [s for s in futures.keys() if futures[s] is not None]:
            for state_2 in [state for state in futures.keys() if state != state_1]:
                if futures[state_2] is not None and self.__compare_futures(futures[state_1], futures[state_2]):
                    if state_2 in futures[state_1][0]: continue  # TODO: Voir si on laisse
                    if timed:
                        # for symbol in set([e.symbol for e in state_1.edges_out]):
                        #     # i_1 = self.__min_max([t for e in state_1.edges_out if e.symbol == symbol for t in e.guard])
                        #     # i_2 = self.__min_max([t for e in state_2.edges_out if e.symbol == symbol for t in e.guard])
                        #     # if self.__overlaps(i_1, i_2):
                        #     #     break  # If at least one pair of edges dosn't overlap, the states won't be splitted again
                        # else:
                        #     continue
                        overlaps = False
                        for edge in state_1.edges_out + state_2.edges_out:
                            i_1 = edge.guard
                            for edge_2 in [e for e in state_1.edges_out + state_2.edges_out if e.symbol == edge.symbol and e != edge]:
                                i_2 = edge_2.guard
                                if self.__overlaps(i_1, i_2):
                                    overlaps = True
                                    break  # If at least one pair of edges overlaps, the states won't be splitted again
                        if not overlaps:
                            continue
                    self.__merge(state_1, state_2, timed)
                    self.__solve_undeterminism(timed)
                    return True
        
        return False

    def __phase_i(self) -> None:
        """
        First phase of the learning process merges operations without considering time
        """
        self.__build_apta()
        merging = True
        while merging:
            merging = self.__look_for_merges(timed=False)

    @staticmethod
    def __assign_interval(i_1: list, i_2: list, tv: int) -> (list, int):
        """
        Assigns an interval to a time value, the interval the time value belongs or the closest interval
        Args:
            i_1 (list): Interval 1
            i_2 (list): Interval 2
            tv (int): Time value
        Returns:
            tuple[list, int]: The assigned interval and the extremity which has to be modified (0=min, 1=max)
        """
        if i_2 < i_1: i_2, i_1 = i_1, i_2  # lexicographical ordering
        if i_1[1] >= tv >= i_1[0]: return i_1, None
        if i_2[1] >= tv >= i_2[0]: return i_2, None
        if tv < i_1[0]: return i_1, 0  # 'min'
        if tv > i_2[1]: return i_2, 1  # 'max'
        if (tv - i_1[1]) < (i_2[0] - tv): return i_1, 1  # 'max'
        return i_2, 0  # 'min'

    def __reallocate_tss(self, edge_1: Edge, new_edge_1: Edge, state_1: State, new_state_1: State, ending: list, interval: list) -> None:
        """
        Assigns each timed string to the good edge in function of the guard after a split
        Args:
            edge_1 (Edge): Resultant edge 1
            new_edge_1 (Edge): Resultant edge 2
            state_1 (State): Destination edge of edge 1
            new_state_1 (State): Destination edge of edge 2
            ending (list): Timed string ending in one of the edges
            interval (list): Guard for edge 2
        """
        tmp = list()
        for ts, ixs in edge_1.tss.items():  # ixs list de de tuple (index, gtime)
            for ix in ixs:  # ix tuple
                tv = eval(self.tss[ts][ix[0]].split(':')[1])
                if ts in ending and ix[0] == len(self.tss[ts]) - 1: continue
                if interval[0] <= tv <= interval[1]:
                    new_edge_1.guard.append(tv)
                    new_edge_1.tss.setdefault(ts, []).append(ix)
                else:
                    tmp.append(tv)
        if len(ending) > 0: state_1.accepting = False
        i_2 = self.__min_max(tmp)
        for ts in ending:
            tv = eval(self.tss[ts][-1].split(':')[1])
            itv, modif = self.__assign_interval(interval, i_2, tv)
            guard = interval if itv == interval else i_2
            if modif is not None: guard[modif] = tv
            if interval[0] <= tv <= interval[1]:
                new_state_1.accepting = True
                i = len(self.tss[ts])-1
                gt = [t[1] for t in edge_1.tss[ts] if t[0] == i][0]
                new_edge_1.tss.setdefault(ts, []).append((i, gt))
            else:
                state_1.accepting = True
        for ts, ixs in new_edge_1.tss.items():
            for ix in ixs:
                edge_1.tss[ts].remove(ix)
                if len(edge_1.tss[ts]) < 1: edge_1.tss.pop(ts)
        new_edge_1.guard = [tv for tv in edge_1.guard if interval[0] <= tv <= interval[1]]
        edge_1.guard = [tv for tv in edge_1.guard if i_2[0] <= tv <= i_2[1]]

    def __split(self, edge_1: Edge, edge_2: Edge, repartition: dict, ending: list) -> None:
        """
        Split an edge
        Args:
            edge_1 (Edge): Edge to split
            edge_2 (Edge): Edge following the edge to split
            repartition (dict): Time value of edge 1 visit in function of edge 2
            ending (list): Timed string ending in one of the edges
        """
        if self.debug:
            print('Splitting')
            edge_1.print()
        # self.ta.save_img(filename="img" + str(self.cpt), bedge=[edge_1], color="darkgreen"); self.cpt += 1
        name = 'S' + str(self.ta.next_state_index())
        state_1 = edge_1.destination
        new_state_1 = self.ta.add_state(name)
        new_state_1.rank = state_1.rank
        new_edge_1 = self.ta.add_edge(edge_1.source.name, new_state_1.name, edge_1.symbol, [])
        edge_2.source = new_state_1
        new_state_1.add_edge(edge_2, 'out')
        state_1.del_edge(edge_2, 'out')
        interval = repartition[edge_2]
        self.__reallocate_tss(edge_1, new_edge_1, state_1, new_state_1, ending, interval)
        self.ta.update_probas()
        self.operations["splits"] += 1
        for s in self.ta.states:
            if len(s.undeterministic_edge_destination(True)) > 0:
                print('Undeterministic automaton.')
        # self.ta.save_img(filename="img" + str(self.cpt), bedge=[edge_1, new_edge_1], color="darkgreen"); self.cpt += 1

    def __look_for_splits(self) -> bool:
        """
        Identifies candidate edges to split
        """
        self.__reorder_states()
        for state in self.ta.states:
            for edge_1 in [e for e in state.edges_out if len(e.destination.edges_out) > 1
                                                         and len(e.destination.edges_in) == 1
                                                         and not e.destination.initial]:  # We can have only one initial state
                repartition = dict.fromkeys(edge_1.destination.edges_out)
                ending = list()
                for ts, ixs in edge_1.tss.items():
                    for ix in [t[0] for t in ixs]:  # t[0]: edge_1 passing index, t[1]: edge_1 passing global time
                        tv_1 = eval(self.tss[ts][ix].split(':')[1])  # edge_1 passing time value
                        if len(self.tss[ts]) == ix + 1: ending.append(ts)
                        else:
                            tuple_2 = self.tss[ts][ix + 1].split(':')
                            symbol, tv_2 = tuple_2[0], eval(tuple_2[1])  # next symbol, next time value
                            edge_2 = self.ta.next_edge(edge_1.destination.name, symbol, tv_2)
                            if repartition[edge_2] is None: repartition[edge_2] = [tv_1, tv_1]
                            repartition[edge_2] = self.__min_max(repartition[edge_2] + [tv_1])
                for edge_2 in repartition.keys():
                    i_1 = repartition[edge_2]
                    others = [item for k, v in repartition.items() if k != edge_2 and v is not None for item in v]
                    i_2 = self.__min_max(others)
                    if not self.__overlaps(i_1, i_2):
                        self.__split(edge_1, edge_2, repartition, ending)
                        return True
        return False

    def __phase_ii(self) -> None:
        """
        Second phase of the learning process with splits and merges operations considering time
        """
        completed = False
        while not completed:
            splitting = self.__look_for_splits()
            for s in self.ta.states:
                if len(s.undeterministic_edge_destination(True)) > 0:
                    print('Undeterministic automaton.')
            if not splitting:
                merging = self.__look_for_merges(True)
                for s in self.ta.states:
                    if len(s.undeterministic_edge_destination(True)) > 0:
                        print('Undeterministic automaton.')
                if not merging:
                    completed = True

    def __import_tss(self, tss_path: str, tss: list) -> list:
        """
        Imports the timed strings for the learning process
        Args:
            tss_path (str): Path to the file containing the timed strings
            tss (list[str]): python list containing the timed strings
        Returns:
            list[str]: List of timed strings
        """
        if tss is None:
            tss_file = open(tss_path)
            tss = tss_file.readlines()
            tss_file.close()
        mem = []
        for ts in tss:
            ts = re.sub('\\n', '', ts)
            ts = ts.split(' ')
            mem.append(ts)
        return mem
