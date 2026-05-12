import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

def tag_dot_to_uppaal_xml(dot_path: str, output_path: str, clock_name: str = "x") -> None:
    """
    Convert a TAG-produced DOT file to a UPPAAL XML file.
    
    Args:
        dot_path: Path to the .gv / .dot file exported by TAG
        output_path: Destination .xml file for UPPAAL
        clock_name: Name of the single clock variable
    """
    # Step 1: Parse DOT file manually
    with open(dot_path) as f:
        dot_content = f.read()

    states, edges, initial_state, accepting_states = parse_dot(dot_content)

    # Step 2: Get positions from Graphviz
    positions = get_graphviz_positions(dot_path, scale=150)

    # Step 3: Build UPPAAL XML
    xml_str = build_uppaal_xml(states, edges, positions, initial_state,
                               accepting_states, clock_name)
    with open(output_path, "w") as f:
        f.write(xml_str)
    print(f"Written to {output_path}")


def parse_dot(dot: str):
    """Extract states and transitions from TAG's DOT format."""
    import re

    # Accepting states have shape=doublecircle
    accepting = set(re.findall(r'(\w+)\s+\[shape="doublecircle"\]', dot))

    # All named nodes (exclude START sentinel)
    all_nodes = set(re.findall(r'\b(S\d+)\b', dot))
    states = all_nodes - {"START"}

    # Initial state is target of START -> X
    init_match = re.search(r'START\s*->\s*(\w+)', dot)
    initial = init_match.group(1) if init_match else "S0"

    # Edges: source -> dest [label="symbol [lo, hi] ..."]
    edge_pattern = re.compile(
        r'(S\d+)\s*->\s*(S\d+)\s*\[label="([^"]+)"\]'
    )
    edges = []
    for m in edge_pattern.finditer(dot):
        src, dst, label = m.group(1), m.group(2), m.group(3)
        # Parse label: "symbol [lo, hi] t[...] p=..."
        parts = label.split()
        symbol = parts[0]
        guard_match = re.search(r'\[(\d+),\s*(\d+)\]', label)
        guard = (int(guard_match.group(1)), int(guard_match.group(2))) if guard_match else (0, 0)
        edges.append((src, dst, symbol, guard))

    return states, edges, initial, accepting


def get_graphviz_positions(dot_path: str, scale: int = 150) -> dict:
    result = subprocess.run(
        ["dot", "-Tplain", dot_path],
        capture_output=True, text=True
    )
    positions = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "node" and parts[1] != "START":
            name = parts[1]
            x = round(float(parts[2]) * scale)
            y = round(float(parts[3]) * scale)
            positions[name] = (x, y)
    return positions


def build_uppaal_xml(states, edges, positions, initial, accepting, clock_name):
    """Produce UPPAAL-compatible XML string."""

    # Build state id map
    state_ids = {s: f"id{i}" for i, s in enumerate(sorted(states))}

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!DOCTYPE nta PUBLIC \'-//Uppaal Team//DTD Flat System 1.6//EN\'',
        '  \'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd\'>',
        '<nta>',
        f'  <declaration>clock {clock_name};</declaration>',
        '  <template>',
        '    <name>TagModel</name>',
        '    <declaration></declaration>',
    ]

    # Locations
    for state in sorted(states):
        sid = state_ids[state]
        x, y = positions.get(state, (0, 0))
        # Graphviz y increases downward in UPPAAL but upward in Graphviz — flip y
        y = -y
        loc_line = f'    <location id="{sid}" x="{x}" y="{y}">'
        lines.append(loc_line)
        lines.append(f'      <name x="{x}" y="{y - 20}">{state}</name>')
        if state in accepting:
            # No direct equivalent; mark as committed as a placeholder
            # or just leave as normal — adjust to your needs
            pass
        lines.append('    </location>')

    # Init
    lines.append(f'    <init ref="{state_ids[initial]}"/>')

    # Transitions
    for (src, dst, symbol, guard) in edges:
        lo, hi = guard
        lines.append('    <transition>')
        lines.append(f'      <source ref="{state_ids[src]}"/>')
        lines.append(f'      <target ref="{state_ids[dst]}"/>')
        # Guard
        guard_expr = f'{clock_name} &gt;= {lo} &amp;&amp; {clock_name} &lt;= {hi}'
        lines.append(f'      <label kind="guard">{guard_expr}</label>')
        # Sync (symbol as channel)
        lines.append(f'      <label kind="synchronisation">{symbol}?</label>')
        # Reset clock on every transition (local time semantics)
        lines.append(f'      <label kind="assignment">{clock_name} = 0</label>')
        lines.append('    </transition>')

    lines += [
        '  </template>',
        '  <system>Process = TagModel(); system Process;</system>',
        '</nta>',
    ]

    return "\n".join(lines)