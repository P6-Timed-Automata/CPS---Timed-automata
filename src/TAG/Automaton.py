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
import subprocess


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
        #print(*mem, sep='\n')
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

    def show(self, p_min: float=0, title: str=None, savePng: bool=False, output_path:str=None) -> None:
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
        #display(Image(s.view()))

        if savePng:
            s = graphviz.Source(tmp)
            os.makedirs(output_path, exist_ok=True)
            file_path_full = os.path.join(output_path, title)
            file_path = s.render(filename=file_path_full, format="png", view=False)
            print("Saved automaton to:", file_path)



    # VERSION with var, global, local
    # def export_ta(self, path: str, symbol_map: dict = None, time: int = 86400, sim_nr: int = 1, data_type = "temp") -> None:
    #     """
    #     Export the automaton as a UPPAAL XML file with Graphviz layout coordinates.
    #
    #     Args:
    #         path (str): Path for the output .xml file
    #         symbol_map (dict, optional): Mapping of symbol names to temperature values,
    #                                      e.g. {'a': 2090, 'b': 2120, 'c': 2148}
    #     """
    #
    #     self.update_probas()
    #
    #     state_ids = {s.name: f"id{i}" for i, s in enumerate(self.states)}
    #     initial = next((s for s in self.states if s.initial), self.states[0])
    #
    #     # Use provided symbol map or fallback
    #     if symbol_map is None:
    #         symbol_values = {sym: i for i, sym in enumerate(self.symbols)}
    #     else:
    #         symbol_values = symbol_map
    #
    #     # ----------------------------
    #     # Graphviz layout generation
    #     # ----------------------------
    #     dot = 'digraph G {\n'
    #     dot += 'START [style=invisible]\n'
    #     dot += 'node [shape="circle"]\n'
    #
    #     for state in self.states:
    #         if state.accepting:
    #             dot += f'{state.name} [shape="doublecircle"]\n'
    #
    #     dot += f'START -> {initial.name}\n'
    #
    #     for state in self.states:
    #         for e in state.edges_out:
    #             dot += f'{e.source.name} -> {e.destination.name} [label="{e.symbol}"]\n'
    #
    #     dot += '}'
    #
    #     positions = {}
    #     try:
    #         result = subprocess.run(
    #             ['dot', '-Tplain'],
    #             input=dot,
    #             capture_output=True,
    #             text=True
    #         )
    #
    #         scale = 400
    #         for line in result.stdout.splitlines():
    #             parts = line.split()
    #             if parts[0] == 'node' and parts[1] != 'START':
    #                 name = parts[1]
    #                 x = round(float(parts[2]) * scale)
    #                 y = round(float(parts[3]) * scale)
    #                 positions[name] = (x, -y)
    #
    #     except FileNotFoundError:
    #         print("Warning: 'dot' command not found. Falling back to grid layout.")
    #
    #     # ----------------------------
    #     # Compute invariants
    #     # ----------------------------
    #     upper_bounds = {}
    #
    #     for state in self.states:
    #         bounds = []
    #
    #         for e in state.edges_out:
    #             _, local_hi = e.reduced_guard()
    #             bounds.append(local_hi)
    #
    #         upper_bounds[state.name] = max(bounds) if bounds else None
    #
    #
    #
    #     # ----------------------------
    #     # Build UPPAAL declarations
    #     # ----------------------------
    #     const_decls = ' '.join(
    #         f'const int {sym} = {val};\n'
    #         for sym, val in symbol_values.items()
    #     )
    #
    #     # Determine initial temp from initial state's outgoing symbol (or custom logic)
    #     initial_symbol = None
    #     if initial.edges_out:
    #         initial_symbol = initial.edges_out[0].symbol
    #
    #     initial_temp_value = initial_symbol if initial_symbol is not None else "0"
    #
    #     lines = []
    #
    #     if data_type == "temp":
    #
    #         lines = [
    #             '<?xml version="1.0" encoding="utf-8"?>',
    #             "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN'",
    #             "  'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>",
    #             '<nta>',
    #             f'  <declaration>clock cl_local, cl_global;\n'
    #             f'{const_decls} \n'
    #             f'int temp = {initial_temp_value};\n'
    #             'const int spike_threshold = 4000;\n'
    #             'int prev_temp;\n'
    #             'bool spike = false;\n'
    #             'bool stable = true;\n'
    #             'const int temp_min = 1800; \n'
    #             'const int temp_max = 2700;\n'
    #             '</declaration>',
    #             '  <template>',
    #             '    <name>TagModel</name>',
    #             '    <declaration></declaration>',
    #         ]
    #     elif data_type == "ecg":
    #         lines = [
    #             '<?xml version="1.0" encoding="utf-8"?>',
    #             "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN'",
    #             "  'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>",
    #             '<nta>',
    #             f'  <declaration>clock cl_local, cl_global;\n'
    #             f'{const_decls} \n'
    #             f'int temp = {initial_temp_value};\n'
    #             'const int flat_threshold = -10; \n'
    #             'const int peak_threshold = 50;\n'
    #             'const int first_flat_window = 95;\n'
    #             'const int second_flat_window = 185;\n'
    #             '</declaration>',
    #             '  <template>',
    #             '    <name>TagModel</name>',
    #             '    <declaration></declaration>',
    #         ]
    #
    #
    #     # ----------------------------
    #     # Locations
    #     # ----------------------------
    #     for i, state in enumerate(self.states):
    #         sid = state_ids[state.name]
    #         x, y = positions.get(state.name, ((i % 10) * 200, (i // 10) * 200))
    #
    #         lines.append(f'    <location id="{sid}" x="{x}" y="{y}">')
    #         lines.append(f'      <name x="{x}" y="{y - 20}">{state.name}</name>')
    #
    #         ub = upper_bounds.get(state.name)
    #         if ub is not None:
    #             lines.append(
    #                 f'      <label kind="invariant" x="{x}" y="{y + 20}">'
    #                 f'cl_local &lt;= {ub}</label>'
    #             )
    #
    #         lines.append('    </location>')
    #
    #     lines.append(f'    <init ref="{state_ids[initial.name]}"/>')
    #
    #     # ----------------------------
    #     # Transitions
    #     # ----------------------------
    #     for state in self.states:
    #         for e in state.edges_out:
    #
    #             # Local timing bounds
    #             local_lo, local_hi = e.reduced_guard()
    #
    #             # Global timing bounds
    #             if len(e.tss) > 0:
    #                 global_lo, global_hi = e.reduce_gtime()
    #             else:
    #                 global_lo, global_hi = local_lo, local_hi  # fallback if no global timing exists
    #
    #             symbol_var = e.symbol
    #
    #             lines.append('    <transition>')
    #             lines.append(f'      <source ref="{state_ids[e.source.name]}"/>')
    #             lines.append(f'      <target ref="{state_ids[e.destination.name]}"/>')
    #
    #             lines.append(
    #                 f'      <label kind="guard">'
    #                 f'cl_local &gt;= {local_lo} &amp;&amp; cl_local &lt;= {local_hi} '
    #                 f'&amp;&amp; '
    #                 f'cl_global &gt;= {global_lo} &amp;&amp; cl_global &lt;= {global_hi}'
    #                 f'</label>'
    #             )
    #
    #
    #             if data_type == "temp":
    #                 lines.append(
    #                     '      <label kind="assignment">'
    #                     f'temp = {symbol_var},\n'
    #                     'spike = ((temp - prev_temp &gt;= spike_threshold) || (prev_temp - temp &gt;= spike_threshold)),\n'
    #                     'stable = ((temp - prev_temp &lt; spike_threshold) || (prev_temp - temp &lt; spike_threshold)) &amp;&amp; (temp &gt;= temp_min) &amp;&amp; (temp &lt;= temp_max),\n'
    #                     f'prev_temp = temp,'
    #                     'cl_local = 0'
    #                     '</label>'
    #                 )
    #             elif data_type == "ecg":
    #                 lines.append(
    #                     f'      <label kind="assignment">'
    #                     f'temp = {symbol_var}, cl_local = 0'
    #                     f'</label>'
    #                 )
    #
    #             lines.append('    </transition>')
    #
    #     # ----------------------------
    #     # Final XML
    #     # ----------------------------
    #     min_temp = min(symbol_values.values())
    #     max_temp = max(symbol_values.values())
    #
    #     temp_bounds_expr = f"temp &gt;= {min_temp} &amp;&amp; temp &lt;= {max_temp}"
    #
    #     max_time = 300
    #
    #     accepting_states = [s.name for s in self.states if s.accepting]
    #
    #     if not accepting_states:
    #         raise ValueError("No accepting/final states defined in automaton.")
    #
    #     final_expr = " || ".join(f"Process.{name}" for name in accepting_states)
    #
    #     if data_type == "temp":
    #         lines += [
    #             '  </template>',
    #             '  <system>Process = TagModel(); system Process;</system>',
    #             '  <queries>',
    #
    #             '    <query>',
    #             f'      <formula>strategy Safe = control: A&lt;&gt; {final_expr}</formula>',
    #             '    </query>',
    #
    #             # Simulation under controller
    #             '    <query>',
    #             f'      <formula>simulate [&lt;={time}; {sim_nr}] {{ temp }} under Safe</formula>',
    #             '    </query>',
    #
    #             # Reachability
    #             '    <query>',
    #             f'      <formula>E&lt;&gt; {final_expr}</formula>',
    #             '    </query>',
    #
    #             # Safety
    #             '    <query>',
    #             '      <formula>A[] not deadlock</formula>',
    #             '    </query>',
    #
    #             # Eventually reach accepting state
    #             '    <query>',
    #             f'      <formula>A&lt;&gt; {final_expr}</formula>',
    #             '    </query>',
    #
    #             # Checks that temperature remains within learned symbolic bounds.
    #             '    <query>',
    #             f'      <formula> A[] ({temp_bounds_expr})</formula>',
    #             '    </query>',
    #
    #             # Expected global clock evolution
    #             '    <query>',
    #             f'      <formula>A[] cl_local &lt;= {max_time}</formula>',
    #             '    </query>',
    #
    #             '    <query>',
    #             f'       <formula>A&lt;&gt; stable under Safe</formula>',
    #             '    </query>',
    #
    #             '    <query>',
    #             f'       <formula>A[] not spike under Safe</formula>',
    #             '    </query>',
    #             '  </queries>',
    #
    #             '</nta>',
    #         ]
    #     elif data_type == "ecg":
    #         for acc in accepting_states:
    #              state_expr = f"Process.{acc}"
    #              lines += [
    #                 '  </template>',
    #                 '  <system>Process = TagModel(); system Process;</system>',
    #                 '  <queries>',
    #
    #                 '    <query>',
    #                 f'      <formula>strategy Safe = control: A&lt;&gt; {state_expr}</formula>',
    #                 '    </query>',
    #
    #                 # Simulation under controller
    #                 '    <query>',
    #                 f'      <formula>simulate [&lt;={time}; {sim_nr}] {{ temp }} under Safe</formula>',
    #                 '    </query>',
    #
    #                 # Reachability
    #                 '    <query>',
    #                 f'      <formula>E&lt;&gt; {state_expr}</formula>',
    #                 '    </query>',
    #
    #                 # Safety
    #                 '    <query>',
    #                 '      <formula>A[] not deadlock</formula>',
    #                 '    </query>',
    #
    #                 # Eventually reach accepting state
    #                 '    <query>',
    #                 f'      <formula>A&lt;&gt; {state_expr} under Safe</formula>',
    #                 '    </query>',
    #
    #                 '    <query>',
    #                 f'       <formula>A&lt;&gt; (temp &gt;= peak_threshold) under Safe</formula>',
    #                 '    </query>',
    #
    #
    #                 '    <query>',
    #                 f'       <formula>A&lt;&gt; (cl_global &lt;= first_flat_window imply temp &lt;= flat_threshold) under Safe</formula>',
    #                 '    </query>',
    #
    #
    #                 '    <query>',
    #                 f'       <formula>A[] (cl_global &lt;= first_flat_window imply temp &lt;= flat_threshold) under Safe</formula>',
    #                 '    </query>',
    #
    #
    #                 '    <query>',
    #                 f'       <formula>A&lt;&gt; (cl_global &gt;= second_flat_window imply temp &lt;= flat_threshold) under Safe </formula>',
    #                 '    </query>',
    #
    #                 '    <query>',
    #                 f'       <formula>A[](cl_global &gt;= second_flat_window imply temp &lt;= flat_threshold) under Safe </formula>',
    #                 '    </query>',
    #
    #                 '  </queries>',
    #
    #
    #                 '</nta>',
    #             ]
    #
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    #     with open(path, 'w+') as f:
    #         f.write('\n'.join(lines))
    #
    #     print(f"UPPAAL model written to {path}")






    def build_declarations(self, symbol_values, initial_symbol, data_type):
        const_decls = ''.join(
            f'const int {sym} = {val};\n'
            for sym, val in symbol_values.items()
        )

        initial_temp_value = initial_symbol if initial_symbol is not None else "0"

        declarations_xml = ""

        if data_type == "temp":
            declarations_xml += (
                '<declaration>\n'
                'clock cl_local, cl_global;\n'
                f'{const_decls} \n'
                f'int temp = {initial_temp_value};\n'
                'int prev_temp;\n'
                'const int spike_threshold = 4000;\n'
                'const int temp_min = 1800; \n'
                'const int temp_max = 2700;\n'
                'const int stabilization_time = 900;\n'
                '</declaration>'
            )
        elif data_type == "ecg":
            declarations_xml += (
                '<declaration>\n'
                'clock cl_local, cl_global;\n'
                f'{const_decls} \n'
                f'int temp = {initial_temp_value};\n'
                'int prev_temp;\n'
                'const int spike_threshold = 20;\n'
                'const int flat_threshold = 6;\n'
                'const int flat_window = 78;\n'
                'const int spike_window = 13;\n'
                '</declaration>'
            )

        return declarations_xml

    def compute_graphviz_positions(self):
        dot = 'digraph G {\n'
        dot += 'START [style=invisible]\n'
        dot += 'node [shape="circle"]\n'

        initial = next((s for s in self.states if s.initial), self.states[0])
        dot += f'START -> {initial.name}\n'

        for state in self.states:
            for e in state.edges_out:
                dot += f'{e.source.name} -> {e.destination.name} [label="{e.symbol}"]\n'

        dot += '}'

        positions = {}

        try:
            result = subprocess.run(
                ['dot', '-Tplain'],
                input=dot,
                capture_output=True,
                text=True
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
            print("Warning: dot not found")

        return positions


    def create_location(self,s_id, pos, name, ub=None):
        if pos is not None:
            x, y = pos
            loc = "\t\t\t<location id=\"id" + str(s_id) + "\" x=\"" + str(round(x)) + "\" y=\"" + str(round(y)) + "\">" + "\n"
            loc += "\t\t\t\t<name x=\"" + str(round(x)) + "\" y=\"" + str(round(y - 20)) + "\">" + name + "</name>" + "\n"
        else:
            loc = "\t\t\t<location id=\"id" + str(s_id) + "\">" + "\n"
            loc += "\t\t\t\t<name>" + name + "</name>" + "\n"
        if ub is not None:
            loc +=  '\t\t\t\t<label kind="invariant">cl_local&lt;='+str(ub)+'</label>\n'
        loc += "\t\t\t</location>" + "\n"
        return loc


    def create_edge(self, e, source, target, proba=False, cl_global=False):
        edge = "\t\t<transition>" + "\n"
        edge += "\t\t\t<source ref=\"" + source + "\"/>" + "\n"
        edge += "\t\t\t<target ref=\"" + target + "\"/>" + "\n"
        if proba:
            edge += "\t\t\t<label kind=\"probability\">" + str(
                round(e.proba * 100)) + "</label>" + "\n"
        else:
            local_lo, local_hi = e.reduced_guard()
            global_lo, global_hi = e.reduce_gtime()

            if cl_global is False:
                edge += (
                    "\t\t\t<label kind=\"guard\">\n"
                    f"cl_local &gt;= {local_lo} &amp;&amp; "
                    f"cl_local &lt;= {local_hi}"
                    "\t\t\t</label>\n"
                )

            else:
                edge += (
                    "\t\t\t<label kind=\"guard\">\n"
                    f"cl_local &gt;= {local_lo} &amp;&amp; "
                    f"cl_local &lt;= {local_hi} &amp;&amp; "
                    f"cl_global &gt;= {global_lo} &amp;&amp; "
                    f"cl_global &lt;= {global_hi}\n"
                    "\t\t\t</label>\n"
                )
            edge += (
                "\t\t\t<label kind=\"assignment\">\n"
                "prev_temp = temp,\n"
                f"temp = {e.symbol},\n"
                "cl_local = 0\n"
                "\t\t\t</label>\n"
            )

        edge += "\t\t</transition>" + "\n"
        return edge





    def build_queries(self, final_expr,  time, sim_nr, data_type = "temp"):
        queries = ""

        if data_type == "temp":
            queries += "\t<queries>\n\n"

            # --------------------------------------
            # SIMULATION TRACE QUERIES
            # --------------------------------------

            # Generate simulation traces for visual inspection of system behavior
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>simulate [&lt;={time}; {sim_nr}] {{ temp }} </formula>\n"
            queries += "\t\t\t<comment>Simulate temperature traces over bounded time horizon to visually inspect spike events and stabilization behavior</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # EXISTENCE / REACHABILITY CHECKS (E<>)
            # --------------------------------------

            # 2. Spike reachable
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>E&lt;&gt; Observer.SPIKE</formula>\n"
            queries += "\t\t\t<comment>Check whether a spike state is reachable in at least one execution path</comment>\n"
            queries += "\t\t</query>\n\n"

            # 4. Stabilization reachable
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>E&lt;&gt; Observer.STABILIZED</formula>\n"
            queries += "\t\t\t<comment>Check whether the system can reach a stabilized state in at least one execution path</comment>\n"
            queries += "\t\t</query>\n\n"

            # Spike followed by stabilization reachable
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>E&lt;&gt; Observer.SPIKE_seen &amp;&amp; Observer.STABILIZED</formula>\n"
            queries += "\t\t\t<comment>Check whether there exists a run where a spike occurs and stabilization is eventually reached afterward</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # SAFETY / STRUCTURAL CONSTRAINTS (A[])
            # --------------------------------------

            # Safety invariant: temperature bounds
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>A[] (temp &gt;= temp_min &amp;&amp; temp &lt;= temp_max)</formula>\n"
            queries += "\t\t\t<comment>Ensure temperature always remains within safe operating bounds in all states</comment>\n"
            queries += "\t\t</query>\n\n"

            # Deadlock freedom
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A[] not deadlock</formula>\n"
            queries += "\t\t\t<comment>Ensure the system is deadlock-free in all reachable states</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # UNIVERSAL REACHABILITY CONSTRAINTS (A<>)
            # --------------------------------------

            # Spike unavoidable (liveness check)
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>A&lt;&gt; Observer.SPIKE</formula>\n"
            queries += "\t\t\t<comment>Check whether spike eventually occurs on all execution paths (liveness requirement)</comment>\n"
            queries += "\t\t</query>\n\n"

            # Stabilization unavoidable
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>A&lt;&gt; Observer.STABILIZED</formula>\n"
            queries += "\t\t\t<comment>Check whether all executions eventually reach a stabilized state</comment>\n"
            queries += "\t\t</query>\n\n"

            # Spike followed by stabilization on all paths
            queries += "\t\t<query>\n"
            queries += "\t\t\t<formula>A&lt;&gt; Observer.SPIKE_seen &amp;&amp; Observer.STABILIZED</formula>\n"
            queries += "\t\t\t<comment>Check whether all executions eventually include both spike occurrence and subsequent stabilization</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # PROBABILISTIC BEHAVIOUR (Pr[])
            # --------------------------------------

            # Probability of spike within bounded time
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}](&lt;&gt; Observer.SPIKE)</formula>\n"
            queries += "\t\t\t<comment>Compute probability that a spike occurs within the given time bound</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability of stabilization within bounded time
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}](&lt;&gt; Observer.STABILIZED)</formula>\n"
            queries += "\t\t\t<comment>Compute probability that the system stabilizes within the given time bound</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability of spike then stabilization
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}](&lt;&gt; (Observer.SPIKE_seen &amp;&amp; Observer.STABILIZED))</formula>\n"
            queries += "\t\t\t<comment>Compute probability that a spike occurs followed by stabilization within the time bound</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t</queries>\n"

        elif data_type == "ecg":

            queries += "\t<queries>\n\n"

            # --------------------------------------
            # SIMULATION TRACE
            # --------------------------------------

            # Generate simulation traces for inspection
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>simulate [&lt;={time}; {sim_nr}] {{ temp }} </formula>\n"
            queries += "\t\t\t<comment>Generate simulation traces of the temperature signal for visual inspection of spike and flat behavior</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # EXISTENCE / REACHABILITY CHECKS (E<>)
            # --------------------------------------

            # Can reach acceptance states
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; {final_expr}</formula>\n"
            queries += "\t\t\t<comment>Check that the model can reach an accepting state</comment>\n"
            queries += "\t\t</query>\n\n"

            # Spike can occur at least once in the model
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.SPIKE</formula>\n"
            queries += "\t\t\t<comment>Verify that at least one spike event is possible</comment>\n"
            queries += "\t\t</query>\n\n"

            # Recovery flat after spike exists
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.FLAT &amp;&amp; Observer.SPIKE_seen</formula>\n"
            queries += "\t\t\t<comment>Verify that the system can return to a flat region after a spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # There exists a flat region before any spike occurs
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.FLAT &amp;&amp; !Observer.SPIKE_seen</formula>\n"
            queries += "\t\t\t<comment>Verify that a flat region can exist before the first spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Initial flat before spike lasts at least flat_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.FLAT &amp;&amp; !Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window</formula>\n"
            queries += "\t\t\t<comment>Check that the initial flat region lasts at least flat_window time units</comment>\n"
            queries += "\t\t</query>\n\n"

            # Recovery flat after spike lasts at least flat_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.FLAT &amp;&amp; Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window</formula>\n"
            queries += "\t\t\t<comment>Check that the recovery flat region after a spike lasts at least flat_window time units</comment>\n"
            queries += "\t\t</query>\n\n"

            # Spike lasts at least spike_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>E&lt;&gt; Observer.SPIKE &amp;&amp; Observer.cl_observer &gt;= spike_window</formula>\n"
            queries += "\t\t\t<comment>Check that a spike can last at least spike_window time units</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # SAFETY / STRUCTURAL CONSTRAINTS (A[])
            # --------------------------------------

            # Deadlock free system
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A[] not deadlock</formula>\n"
            queries += "\t\t\t<comment>Ensure that the model is deadlock-free in all reachable states</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"

            # --------------------------------------
            # UNIVERSAL REACHABILITY CONSTRAINS (A<>)
            # --------------------------------------

            # Spike eventually occurs in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.SPIKE</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually reaches a spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Full cycle of FLAT->SPIKE->FLAT in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.FLAT &amp;&amp; Observer.SPIKE_seen</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually goes through a spike and reaches a recovery flat</comment>\n"
            queries += "\t\t</query>\n\n"

            # Initial flat occurs before spike in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.FLAT &amp;&amp; !Observer.SPIKE_seen</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually reaches an initial flat before spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Initial flat lasts enough time in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.FLAT &amp;&amp; !Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually has initial flat lasting at least flat_window</comment>\n"
            queries += "\t\t</query>\n\n"

            # Recovery flat lasts enough time in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.FLAT &amp;&amp; Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually has recovery flat lasting at least flat_window</comment>\n"
            queries += "\t\t</query>\n\n"

            # Spike lasts enough time in all executions
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>A&lt;&gt; Observer.SPIKE &amp;&amp; Observer.cl_observer &gt;= spike_window</formula>\n"
            queries += "\t\t\t<comment>Every execution eventually has a spike lasting at least spike_window</comment>\n"
            queries += "\t\t</query>\n\n"

            queries += "\t\t<query>\n"
            queries += "\t\t</query>\n\n"



            # --------------------------------------
            # PROBABILISTIC BEHAVIOUR (Pr[])
            # --------------------------------------

            # Probability of a spike occurring within simulation
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.SPIKE)) </formula>\n"
            queries += "\t\t\t<comment>Probability that at least one spike occurs within the simulation time</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability of full FLAT->SPIKE->FLAT cycle
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.FLAT &amp;&amp; Observer.SPIKE_seen)) </formula>\n"
            queries += "\t\t\t<comment>Probability that a full cycle (flat->spike->flat) occurs</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability of initial flat before spike
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.FLAT &amp;&amp; !Observer.SPIKE_seen)) </formula>\n"
            queries += "\t\t\t<comment>Probability that an initial flat region occurs before any spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability that initial flat lasts at least flat_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.FLAT &amp;&amp; Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window)) </formula>\n"
            queries += "\t\t\t<comment>Probability that flat region lasts at least flat_window time units after a spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability that initial flat lasts at least flat_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.FLAT &amp;&amp; !Observer.SPIKE_seen &amp;&amp; Observer.cl_observer &gt;= flat_window)) </formula>\n"
            queries += "\t\t\t<comment>Probability that flat region lasts at least flat_window time units before a spike</comment>\n"
            queries += "\t\t</query>\n\n"

            # Probability that spike lasts at least spike_window
            queries += "\t\t<query>\n"
            queries += f"\t\t\t<formula>Pr[&lt;={time}; {sim_nr}] (&lt;&gt; (Observer.SPIKE &amp;&amp; Observer.cl_observer &gt;= spike_window)) </formula>\n"
            queries += "\t\t\t<comment>Probability that spike lasts at least spike_window time units</comment>\n"
            queries += "\t\t</query>\n\n"


            queries += "\t</queries>\n"

        return queries


    def export_ta(self, ta, path, symbol_map=None, time=86400, sim_nr=1, data_type="temp"):
        s_id = 0
        d_s_id = dict.fromkeys([s.name for s in ta.states] + ta.symbols)
        xml = ""
        branchpoint = ""
        init = ""
        trans_ini = ""

        initial = next((s for s in self.states if s.initial), self.states[0])

        initial_symbol = (
            initial.edges_out[0].symbol
            if initial.edges_out
            else "0"
        )

        declarations = self.build_declarations(symbol_map, initial_symbol, data_type)

        xml += '<?xml version="1.0" encoding="utf-8"?>' + '\n'
        xml += "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>" + '\n'
        xml += '<nta>' + '\n'
        xml += f'{declarations}' + '\n'

        xml += '\t<template>' + '\n'
        xml += '\t\t<name x="5" y="5">TagModel</name>' + '\n'
        xml += '\t\t<declaration>' + '\n'
        xml += '\t\t</declaration>' + '\n'


        positions = self.compute_graphviz_positions()
        # States
        for state in ta.states:
            if state.initial:
                init = '\t\t\t<init ref="'+'id'+str(s_id)+'"/>' + "\n"
                if len(state.edges_out) > 1:
                    xml += "\t\t\t<location id=\"id" + str(s_id) + "\">" + "\n"
                    xml += "\t\t\t\t<name>INIT</name>\n\t\t\t\t<urgent/>\n\t\t\t</location>\n"

                    trans_ini += "\t\t<transition>" + "\n"
                    trans_ini += "\t\t\t<source ref=\"id" + str(s_id) + "\"/>" + "\n"
                    s_id += 1
                    trans_ini += "\t\t\t<target ref=\"id" + str(s_id) + "\"/>" + "\n"
                    trans_ini += "\t\t</transition>" + "\n"


            if len(state.edges_out) < 2:
                ub = max(next(iter(state.edges_out)).guard) if state.edges_out else None
                pos = positions.get(state.name)
                xml += self.create_location(s_id, pos, state.name, ub)
                d_s_id[state.name] = "id" + str(s_id)
                s_id += 1
            else:
                branchpoint += "\t\t\t<branchpoint id=\"id" + str(s_id) + "\">" + "\n"
                d_s_id[state.name] = "id" + str(s_id)
                branchpoint += "\t\t\t</branchpoint>" + "\n"
                s_id += 1

                for e in state.edges_out:
                    ub = max(e.guard)
                    name = state.name + '_' + e.destination.name + '_' + e.symbol
                    pos = positions.get(state.name)
                    xml += self.create_location(s_id, pos, name, ub)
                    d_s_id[name] = "id" + str(s_id)
                    s_id += 1


        xml += branchpoint + init + trans_ini

        # Transitions
        for state in ta.states:
            if len(state.edges_out) == 1:
                e = state.edges_out.pop()
                xml += self.create_edge(e, d_s_id[e.source.name], d_s_id[e.destination.name])
            else:
                for e in state.edges_out:
                    name = state.name + '_' + e.destination.name + '_' + e.symbol
                    # Transition proba
                    xml += self.create_edge(e, d_s_id[state.name], d_s_id[name], proba=True)
                    # Transition guard
                    xml += self.create_edge(e, d_s_id[name], d_s_id[e.destination.name])

        # Connect acceptance states back to init
        initial_id = "id0"

        for state in ta.states:
            if not state.accepting:
                continue

            if state.accepting:
                source_id = d_s_id.get(state.name)
                # Create a transition back to the initial state
                # add invariant
                pattern = rf'(<location id="{source_id}"[^>]*>\s*<name[^>]*>[^<]*</name>)'
                replacement = r'\1\n\t\t\t\t<label kind="invariant">cl_local &lt;= 0</label>'
                xml = re.sub(pattern, replacement, xml, count=1)


                xml += "\t\t<transition>\n"
                xml += f"\t\t\t<source ref=\"{source_id}\"/>\n"
                xml += f"\t\t\t<target ref=\"{initial_id}\"/>\n"
                # Assignment: reset local clock
                xml += "\t\t\t<label kind=\"assignment\">cl_local = 0</label>\n"
                xml += "\t\t</transition>\n"



        xml += "\t</template>"

        # Observer
        xml += "\t<template>\n"
        xml += "\t\t<name>TagObserver</name>\n"
        xml += "\t\t<declaration>\n"
        xml += "\t\t\tclock cl_observer;\n"
        xml += "\t\t\tbool SPIKE_seen = false;\n"
        xml += "\t\t</declaration>\n"

        # --- Auxiliary automaton for verification ---
        if data_type == "temp":
            # States
            xml += self.create_location(s_id, (0,0), "START")
            d_s_id["START"] = "id" + str(s_id)
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 0</label>\n'
            s_id += 1

            xml += self.create_location(s_id, (200,50), "SPIKE")
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 90000</label>\n'
            d_s_id["SPIKE"] = "id" + str(s_id)
            s_id += 1

            xml += self.create_location(s_id, (400,-50), "STABLE")
            d_s_id["STABLE"] = "id" + str(s_id)
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= stabilization_time</label>\n'
            s_id += 1

            xml += self.create_location(s_id, (600,100), "STABILIZED")
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 90000</label>\n'
            d_s_id["STABILIZED"] = "id" + str(s_id)
            s_id += 1

            # Initial state
            xml += f'\t\t<init ref="{d_s_id["START"]}"/>\n'

            # Transitions
            # START -> SPIKE
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["START"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["SPIKE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">((temp - prev_temp &gt;= spike_threshold) || (prev_temp - temp &gt;= spike_threshold))</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0,SPIKE_seen = true</label>\n'
            xml += "\t\t</transition>\n"

            # START -> STABLE (if first value is already stable)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["START"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["STABLE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">((temp - prev_temp &lt; spike_threshold) &amp;&amp; (temp &gt;= temp_min) &amp;&amp; (temp &lt;= temp_max))</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            # SPIKE -> STABLE
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["SPIKE"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["STABLE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">((temp - prev_temp &lt; spike_threshold) &amp;&amp; (temp &gt;= temp_min) &amp;&amp; (temp &lt;= temp_max))</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            # STABLE -> STABILIZED (after some time)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["STABLE"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["STABILIZED"]}"/>\n'
            xml += '\t\t\t<label kind="guard">cl_observer &gt;= stabilization_time</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            # STABLE -> SPIKE (allow another spike before stabilization)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["STABLE"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["SPIKE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">((temp - prev_temp &gt;= spike_threshold) || (prev_temp - temp &gt;= spike_threshold))</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            # STABILIZED -> SPIKE (allow spikes after stabilization)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["STABILIZED"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["SPIKE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">((temp - prev_temp &gt;= spike_threshold) || (prev_temp - temp &gt;= spike_threshold))</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            xml += "\t</template>\n"


        elif data_type == "ecg":
            # START
            xml += self.create_location(s_id, (0, 0), "START")
            d_s_id["START"] = "id" + str(s_id)
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 0</label>\n'
            s_id += 1

            # FLAT (loop location)
            xml += self.create_location(s_id, (200, 50), "FLAT")
            d_s_id["FLAT"] = "id" + str(s_id)
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 300</label>\n'
            s_id += 1

            # SPIKE
            xml += self.create_location(s_id, (400, -50), "SPIKE")
            d_s_id["SPIKE"] = "id" + str(s_id)
            xml += f'\t\t<label kind="invariant">cl_observer &lt;= 300</label>\n'
            s_id += 1

            # Initial state
            xml += f'\t\t<init ref="{d_s_id["START"]}"/>\n'


            # Transitions
            # START -> FLAT (if first value is already flat)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["START"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["FLAT"]}"/>\n'
            xml += '\t\t\t<label kind="guard">(temp &lt;= flat_threshold)</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0, SPIKE_seen = false</label>\n'
            xml += "\t\t</transition>\n"

            # FLAT -> SPIKE
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["FLAT"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["SPIKE"]}"/>\n'
            xml += '\t\t\t<label kind="guard">(temp &gt;= spike_threshold)</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0, SPIKE_seen = true</label>\n'
            xml += "\t\t</transition>\n"

            # SPIKE -> FLAT (return to flat, loop back)
            xml += "\t\t<transition>\n"
            xml += f'\t\t\t<source ref="{d_s_id["SPIKE"]}"/>\n'
            xml += f'\t\t\t<target ref="{d_s_id["FLAT"]}"/>\n'
            xml += '\t\t\t<label kind="guard">(temp &lt;= flat_threshold)</label>\n'
            xml += '\t\t\t<label kind="assignment">cl_observer = 0</label>\n'
            xml += "\t\t</transition>\n"

            xml += "\t</template>\n"


        xml += '\t<system>' + "\n"
        xml += 'Automaton = TagModel();' + "\n"
        xml += 'Observer = TagObserver();' + "\n"
        xml += 'system Automaton, Observer;' + "\n"
        xml += '\t</system>' + "\n"



        # --- queries ---
        final_expr = " || ".join(f"Automaton.{s.name}" for s in self.states if s.accepting)

        xml += self.build_queries(final_expr, time, sim_nr, data_type)

        xml += '</nta>'

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)

        print(f"UPPAAL model written to {path}")




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


    def evaluate_classifier(self,
                            positive_tss: list,
                            negative_tss: list,
                            timed: bool = True,
                            save_path: str = None,
                            run_id: str = "run") -> dict:
        """
        Evaluate automaton as binary classifier using positive and negative traces.
        Returns precision, recall, F1, accuracy, confusion matrix.
        """

        n_positive = len(positive_tss)
        n_negative = len(negative_tss)

        TP = FN = FP = TN = 0

        # Positive samples
        for ts in positive_tss:
            if self.__exist_path(ts, timed):
                TP += 1
            else:
                FN += 1

        # Negative samples
        fp_indices = []
        for i,ts in enumerate(negative_tss):
            if self.__exist_path(ts, timed):
                FP += 1
                fp_indices.append(i)
            else:
                TN += 1

        # Metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        positive_acceptance_rate = 100 * TP / n_positive
        negative_acceptance_rate = 100 * FP / n_negative


        metrics = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "PAR": positive_acceptance_rate,
            "NAR": negative_acceptance_rate,
        }

        # Save log
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file_exists = os.path.exists(save_path)

            with open(save_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write(
                        "run_id,NR_POS,NR_NEG,TP,FP,TN,FN,precision,recall,f1,PAR,NAR,FP_indices\n"
                    )

                f.write(
                    f"{run_id},"
                    f"{n_positive},{n_negative},"
                    f"{TP},{FP},{TN},{FN},"
                    f"{precision:.4f},"
                    f"{recall:.4f},"
                    f"{f1:.4f},"
                    f"{positive_acceptance_rate:.4f},"
                    f"{negative_acceptance_rate:.4f},"
                    f"{fp_indices}\n"
                )

        return metrics


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