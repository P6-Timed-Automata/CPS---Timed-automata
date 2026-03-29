import re
from typing import Union
try:
    import graphviz
    from IPython.display import Image, display
    import tempfile
    has_graphviz = True
except ImportError:
    has_graphviz = False

from TAG.Edge import Edge
from TAG.State import State
import os


class Automaton:
    """
    An instance of the class Automaton is a Timed Automaton
    Attributes:
        states (list[State]): list of states of the automaton
        edges (list[Edge]): list of edges of the automaton
        symbols (list[str]): alphabet if the automaton
    """
    def __init__(self, dot_path:str=None):
        """
        Create an automaton with an initial state named 'S0' if no dot path, create an automaton from a dot file otherwise
        Args:
            dot_path (:obj:`str`, optional): Path to an automaton in DOT format
        """
        self.kfutures = {}
        self.states = []
        self.edges = []
        self.symbols = []
        if dot_path is None:
            self.add_state('S0', initial=True)
        else:
            self.import_from_dot(dot_path)
        self.tss = []

    def save_img(self, filename="test", path="", bedge=None, bstate=None, color="dodgerblue4"):
        """
        Save the timed automaton image \n
        Args:
            filename (str): Name of the file to save
            path (str): Path where the save the file
            bedge (list[Edge], optional): List of edges to color and print in bold
            bstate (list[State], optional): List of states to color and print in bold
            color (str, optional): Color for states and edges in bedge and bstate
        """
        if not has_graphviz: return
        self.update_probas()
        tmp = 'digraph G {\n' + 'START [style=invisible]\n'
        tmp += 'graph [fontname = "helvetica"]\n'
        tmp += 'node [fontname = "helvetica", shape="circle"]\n'
        tmp += 'edge [fontname = "helvetica"]\n'
        for state in self.states:
            if bstate is not None and state in bstate:
                tmp += state.name + ' [penwidth=3, fontname="helvetica bold", color=' + color + ', fontcolor=' + color + ']\n'
            if state.accepting:
                tmp += state.name + ' [shape="doublecircle"]\n'
        tmp += 'START -> S0\n'
        for edge in [e for e in self.edges if e.source is not None and e.destination is not None]:
            tmp += edge.source.name + ' -> ' + edge.destination.name + ' [label="' + edge.symbol + ' ' + str(edge.reduced_guard())
            tmp += '\\nt[' + str(edge.reduce_gtime()[0]) + ', ' + str(edge.reduce_gtime()[1]) + ']\\np=' + str(round(edge.proba, 2)) + '"'
            if bedge is not None and edge in bedge:
                tmp += ', penwidth=3, fontname="helvetica bold", color=' + color + ', fontcolor=' + color + ']\n'
            else: tmp += ']\n'
        tmp += '}'
        f = open(path+filename+".txt", "w+")
        f.writelines(tmp)
        f.close()
        s = graphviz.Source(tmp, filename=path+filename+".gv", format="png")
        s.render()

    def update_probas(self) -> None:
        """
        Update the edges probability of access
        """
        for state in self.states:
            sum = 0
            for edge in state.edges_out:
                sum += edge.visit_number()
            for edge in state.edges_out:
                edge.proba = edge.visit_number() / sum

    def add_state(self, name:str, accepting:bool=False, initial:bool=False) -> State:
        """
        Create and add a new state to the state list of the automaton \n
        Args:
            name (str): Name of the new state
            accepting (:obj:`bool`, optional): True if the state is accepting
            initial (:obj:`bool`, optional): True if the state is initial
        Returns:
            State: The added state
        """
        s = State(name, initial, accepting)
        self.states.append(s)
        return s

    def add_edge(self, source: str, destination: str, symbol: str, guard: list) -> Edge:
        """
        Create and add a new edge to the edge list of the automaton \n
        Args:
            source (str): State name of the source of the edge
            destination (str): State name of the destination of the edge
            symbol (str): Symbol of the edge
            guard (list[int]): List of possible time values for the edge
        Returns:
            Edge: The added edge
        """
        if source not in [state.name for state in self.states]:
            source = self.add_state(source)
        else:
            i = [state.name for state in self.states].index(source)
            source = self.states[i]
        if destination not in [state.name for state in self.states]:
            destination = self.add_state(destination)
        else:
            i = [state.name for state in self.states].index(destination)
            destination = self.states[i]
        e = Edge(source, destination, symbol, guard)
        self.edges.append(e)
        return e

    def search_state(self, name: str) -> Union[State, None]:
        """
        Search the state of the automaton having a specific name \n
        Args:
            name (str): Name of the researched state
        Returns:
            Union[State, None]: The state having the specified name, nothing if not found
        """
        d = {s.name: s for s in self.states}
        if name in d.keys(): return d[name]
        else: return None

    def next_edge(self, last: str, symbol: str, time_value: int = None) -> Union[Edge, None]:
        """
        Search the edge accessible from a given state, with a given symbol and a given time value (optional) \n
        Args:
            last (str): name of the source state of the researched transition
            symbol (str): symbol of the researched transition
            time_value (:obj:`int`, optional): Optional, the time value acceptable for the researched transition
        Returns:
            Union[Edge, None]: The edge accessible, nothing if none
        """
        source = self.search_state(last)
        for e in source.edges_out:
            if e.symbol == symbol:
                if time_value is not None:
                    if min(e.guard) <= time_value <= max(e.guard): return e
                else: return e

    def next_state_index(self) -> int:
        """
        Returns:
            int: The smallest state index available
        """
        liste = []
        available = False
        for state in self.states:
            liste.append(eval(state.name[1:]))  # pas le 'S'
        i = 0
        while not available:
            i += 1
            if i not in liste: available = True
        return i

    def print(self, reduced_guard=True, gtime=True) -> list:
        """
        Print the transitions of the automaton in the dot syntax
        SOURCE_STATE -> DESTINATION_STATE [label='SYMBOL GUARD p=PROBABILITY'] \n
        Args:
            reduced_guard (:obj:`bool`, optional): False if all the time values encountered during learning should be printed, true (default) to only print interval.
            gtime (:obj:`bool`, optional): True if the global clock should be displayed, True by default.
        Returns:
            list[str]: A list where each element is a line of the dot file
        """
        mem = []
        for state in self.states:
            for e in state.edges_out:
                tmp = e.source.name + ' -> ' + e.destination.name
                tmp += ' [label="' + e.symbol + ' '
                if reduced_guard:
                    tmp += str(e.reduced_guard()) + ' '
                else:
                    tmp += str(e.guard) + ' '
                if gtime:
                    if len(e.tss) > 0:
                        gtime = e.reduce_gtime()
                        tmp += "t[" + str(gtime[0]) + ", " + str(gtime[1]) + "]" + ' '
                tmp += 'p=' + str(round(e.proba, 2)) + '"]'
                mem.append(tmp)
        print(*mem, sep='\n')
        return mem

    def print_p(self, p_min:float, mem:set=set(), state:str='S0', states:set={'S0'}, global_time=False) -> tuple:
        """
        Recursively build the strings to print the transitions having a minimal probability of access \n
        Args:
            p_min (float): Minimal probability of the printed edges
            mem (:obj:`set`, optional): Memory for the recursive process
            state (:obj:`str`, optional): Current state for recursion
            states (:obj:`str`, optional): Visited states for recursion
            global_time (:obj:`bool`, optional): True if the global clock should be displayed, False by default.
        Returns:
            tuple[set[str], set[str]]: The first component is a set of strings of the transitions and the second component is a set of state names to print
        """
        state = self.search_state(state)
        for edge in state.edges_out:
            if edge.proba >= p_min:
                if edge.source.name not in states: states.add(edge.source.name)
                if edge.destination.name not in states: states.add(edge.destination.name)
                tmp = edge.source.name + ' -> ' + edge.destination.name
                tmp += ' [label="' + edge.symbol + ' '
                tmp += str(edge.reduced_guard()) + ' '
                if len(edge.tss) > 0 and global_time:
                    gtime = edge.reduce_gtime()
                    tmp += "t[" + str(gtime[0]) + ", " + str(gtime[1]) + "]" + ' '
                tmp += 'p=' + str(round(edge.proba, 2)) + '"]'
                if tmp not in mem:
                    mem.add(tmp)
                    mem, states = self.print_p(p_min, mem, edge.destination.name, states)
                else:
                    return (mem, states)
        return (mem, states)

    def show(self, p_min: float=0, title: str=None, savePng: bool=False) -> None:
        """
        Create a temporary file of the automaton graph \n
        Args:
            p_min (:obj:`float`, optional): minimal probability of access for a path to be printed, 0 by default
            title (:obj:`str`, optional): optional, title of the automaton
        """
        if not has_graphviz: return
        tmp = 'digraph G {\n' + 'START [style=invisible]\n'
        tmp += 'graph [fontname = "helvetica"]\n'
        tmp += 'node [fontname = "helvetica"]\n'
        tmp += 'edge [fontname = "helvetica"]\n'
        if title is not None:
            tmp += 'labelloc="t"\nlabel="' + title + '"\n'
        mem, states = self.print_p(p_min, mem=set(), state='S0', states={'S0'})
        if len(states) > 200:
            print('TA too large. (', str(len(states)), 'states)')
            print(mem)
            return
        for state in states:
            s = self.search_state(state)
            if s.accepting:
                tmp += s.name + ' [shape="doublecircle"]\n'
            else:
                tmp += s.name + ' [shape="circle"]\n'
        tmp += 'START -> S0\n'
        mem = self.print()
        for line in mem:
            tmp += line + '\n'
        tmp += '}'
        s = graphviz.Source(tmp, filename=tempfile.mktemp('.gv'), format="png")
        display(Image(s.view()))

        if savePng:
            output_path = os.path.join("src", "Results", title)
            os.makedirs(os.path.join("src", "Results"), exist_ok=True)
            s = graphviz.Source(tmp)
            file_path = s.render(filename=output_path, format="png", view=False)
            print("Saved automaton to:", file_path)

    def export_ta(self, path: str, symbol_map: dict = None) -> None:
        """
        Export the automaton as a UPPAAL XML file with Graphviz layout coordinates.
        Args:
            path (str): Path for the output .xml file
            symbol_map (dict, optional): Mapping of symbol names to integer values for plotting,
                                         e.g. {'a': 2090, 'b': 2120, 'c': 2148}.
                                         If None, symbols are mapped to indices 0, 1, 2, ...
        """
        import subprocess

        self.update_probas()

        state_ids = {s.name: f"id{i}" for i, s in enumerate(self.states)}
        initial = next((s for s in self.states if s.initial), self.states[0])

        # Use provided symbol map or fall back to integer indices
        if symbol_map is None:
            symbol_values = {sym: i for i, sym in enumerate(self.symbols)}
        else:
            symbol_values = symbol_map

        # Build DOT string to feed into Graphviz for layout
        dot = 'digraph G {\n'
        dot += 'START [style=invisible]\n'
        dot += 'node [shape="circle"]\n'
        for state in self.states:
            if state.accepting:
                dot += f'{state.name} [shape="doublecircle"]\n'
        dot += 'START -> S0\n'
        for state in self.states:
            for e in state.edges_out:
                dot += f'{e.source.name} -> {e.destination.name} [label="{e.symbol}"]\n'
        dot += '}'

        # Run dot -Tplain to extract positions
        positions = {}
        try:
            result = subprocess.run(
                ['dot', '-Tplain'],
                input=dot, capture_output=True, text=True
            )
            scale = 400
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts[0] == 'node' and parts[1] != 'START':
                    name = parts[1]
                    x = round(float(parts[2]) * scale)
                    y = round(float(parts[3]) * scale)
                    positions[name] = (x, -y)
        except FileNotFoundError:
            print("Warning: 'dot' command not found. Falling back to grid layout.")

        # Compute invariant upper bounds
        upper_bounds = {}
        for state in self.states:
            bounds = [e.reduced_guard()[1] for e in state.edges_out]
            upper_bounds[state.name] = max(bounds) if bounds else None

        # Build const int declarations for each symbol
        const_decls = ' '.join(
            f'const int {sym} = {val};' for sym, val in symbol_values.items()
        )

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN'",
            "  'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>",
            '<nta>',
            f'  <declaration>clock t; int temp; {const_decls}</declaration>',
            '  <template>',
            '    <name>TagModel</name>',
            '    <declaration></declaration>',
        ]

        for i, state in enumerate(self.states):
            sid = state_ids[state.name]
            x, y = positions.get(state.name, ((i % 10) * 200, (i // 10) * 200))
            lines.append(f'    <location id="{sid}" x="{x}" y="{y}">')
            lines.append(f'      <name x="{x}" y="{y - 20}">{state.name}</name>')
            ub = upper_bounds.get(state.name)
            if ub is not None:
                lines.append(f'      <label kind="invariant" t="{x}" y="{y + 20}">t &lt;= {ub}</label>')
            lines.append('    </location>')

        lines.append(f'    <init ref="{state_ids[initial.name]}"/>')

        for state in self.states:
            for e in state.edges_out:
                lo, hi = e.reduced_guard()
                val = symbol_values.get(e.symbol, 0)
                lines.append('    <transition>')
                lines.append(f'      <source ref="{state_ids[e.source.name]}"/>')
                lines.append(f'      <target ref="{state_ids[e.destination.name]}"/>')
                lines.append(f'      <label kind="guard">t &gt;= {lo} &amp;&amp; t &lt;= {hi}</label>')
                lines.append(f'      <label kind="assignment">temp = {val}, t = 0</label>')
                lines.append('    </transition>')

        lines += [
            '  </template>',
            '  <system>Process = TagModel(); system Process;</system>',
            '  <queries>',
            '    <query>',
            '      <formula>simulate [&lt;=18000] { temp }</formula>',
            '      <comment/>',
            '    </query>',
            '  </queries>',
            '</nta>',
        ]

        with open(path, 'w+') as f:
            f.write('\n'.join(lines))

    def import_from_dot(self, dot_path: str) -> None:
        """
        Create an Automaton instance from a DOT file
        Args:
            dot_path (str): Path to the automaton DOT file
        """
        dot_file = open(dot_path)
        lines = dot_file.readlines()
        dot_file.close()
        for line in lines:
            if re.search('^//', line) is not None: continue
            line = re.sub('//.*', '', line)
            if re.search('->', line) is None: continue
            if re.search('label', line) is None: continue
            line = re.sub(r"^\s+", "", line)  # remove space at the beginning
            m = re.search('^[\w]+(?=\s*)', line)
            source = str(m.group(0))
            m = re.search('(?<=-> )[\w]+', line)
            destination = str(m.group(0))
            m = re.search('(?<=")[\w\?\!]+', line)
            symbol = str(m.group(0))
            if symbol not in self.symbols: self.symbols.append(symbol)
            m = re.search('(?<=\[)(([\d]+, )?)+[\d]+(?=\])', line)
            res = eval(m.group(0))
            if isinstance(res, int): guard = [res]
            else: guard = list(res)
            self.add_edge(source, destination, symbol, guard)
        self.search_state('S0').initial = True

    def __exist_path(self, ts: list, timed: bool, initial: str = 'S0') -> bool:
        """
        Tests if there is a path in the automaton consistent with the timed string
        Args:
            ts (list[str]): Timed string to test
            timed (bool): True the time values must be taken into consideration
            initial (:obj:`str`, optional): Name of the state where to start the path, S0 by default
        Returns:
            bool: True if there is a path, False otherwise
        """
        seq_edges = []
        last = self.search_state(initial)
        seq_states = [last]
        for pair in ts[:-1]:
            pair = pair.split(':')
            if timed:
                edge = self.next_edge(last.name, pair[0], eval(pair[1]))
            else:
                edge = self.next_edge(last.name, pair[0])
            if edge is None: return False
            last = edge.destination
            seq_edges.append(edge)
            seq_states.append(last)
        pair = ts[-1].split(':')
        if timed:
            edge = self.next_edge(last.name, pair[0], eval(pair[1]))
        else:
            edge = self.next_edge(last.name, pair[0])
        if edge is None: return False
        last = edge.destination
        seq_edges.append(edge)
        seq_states.append(last)
        return True

    def inconsistency_nb(self, tss: list, timed: bool, show: bool = True, p: bool = True) -> int:
        """
        Tests if the automaton is consistent with a set of timed strings
        Args:
            tss (list[str]): List of timed strings
            timed (bool): True if time values should be taken into consideration
            show (:obj:`bool`, optional): True if the automaton should be displayed if an inconsistency is found
            p (:obj:`bool`, optional): True if the timed string should be printed if an inconsistency is found
        Returns:
            int: Number of timed strings inconsistent with the automaton
        """
        mem = list()
        for ts in tss:
            exist = self.__exist_path(ts, timed)
            if not exist:
                mem.append(tss.index(ts))
        if len(mem) > 0:
            if p:
                for ts in mem:
                    print(tss[ts])
            if show: self.show()
        return len(mem)

    def show_h(self, state: State, text: str = "") -> None:
        """
        Displays the automaton with a state highlighted
        Args:
            state (State): State to highlight
            text (:obj:`str`, optional): A text to add next to the automaton
        """
        tmp = 'digraph G {\n' + 'START [style=invisible]\n'
        tmp += 'graph [fontname = "helvetica"]\n'
        tmp += 'node [fontname = "helvetica"]\n'
        tmp += 'edge [fontname = "helvetica"]\n'
        tmp += state.name + ' [fillcolor=yellow, style=filled]\n'
        tmp += 'text [shape=box, label="' + text + '"]\n'
        mem, states = self.print_p(0, mem=set(), state='S0', states={'S0'})
        if len(states) > 200:
            print('TA too large. (', str(len(states)), 'states)')
            print(mem)
            return
        for state in states:
            s = self.search_state(state)
            if s.accepting:
                tmp += s.name + ' [shape="doublecircle"]\n'
            else:
                tmp += s.name + ' [shape="circle"]\n'
        tmp += 'START -> S0\n'
        mem = self.print()
        for line in mem:
            tmp += line + '\n'
        tmp += '}'
        s = graphviz.Source(tmp, filename=tempfile.mktemp('.gv'), format="png")
        display(Image(s.view()))