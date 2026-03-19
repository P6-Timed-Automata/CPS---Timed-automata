# &#127991; TAG: Timed Automaton Generator

Passive learner of Timed Automata from event logs.

## Overview
TAG is an algorithm to learn a global, interpretable, and acurate model of a time dependent system from event logs without any a-priori knowledge about it.
It uses the formalism of Timed Automata.
See following publications for more information:
- TAG: Learning of Timed Automata from logs. Lénaïg Cornanguer, Christine Largouët, Laurence Rozé, Alexandre Termier. AAAI 2022. 
- Passive learning of Timed Automata from logs (Student Abstract). Lénaïg Cornanguer. AAAI 2021. 

## Usage

### &#128221; Input
The input file containing the event logs must be formatted as following:
```
event:delay event:delay ... event:delay
...
event:delay event:delay ... event:delay
```
The input sequences can also be provided in a Python list of strings with one sequence per string in the same format.

### &#128187; Example

```python
from src.TAG.TALearner import TALearner

learner = TALearner(filename, k=3)
learner.ta.show()

tss = ["a:1 b:1 c:3", "a:2 b:1 c:1", "a:1 c:2"]
learner = TALearner(tss_list=tss)
learner.ta.print()
```

### &#9881; Arguments
* tss_path (str, optional): path of the file containing the timed strings (must be specified if no tss_list)
* tss_list (list[str], optional): python list containing the timed strings (must be specified if no tss_path)
* k (int, default=2): number of transitions to consider for the merging process, 2 by default
* res_path (str, optional): if an export is required, path to the file where to export the learned automaton
* display (bool, optional): true if the learned automaton should be displayed at the end, false by default
* debug (boolean, default=False): true for verbose mode
* splits (boolean, default=True): true if splits are allowed
* order (str, default=breadth-first): ordering method for the operations (breadth-first/depth-first/random/bottom-up)

### &#10024; Output
* A temporary PNG file of the resulting timed automaton if less than 200 states and no export required (requires Graphviz)
* The resulting automaton in DOT format printed in the console if no export required
* The resulting automaton in DOT format in res_path file if export required (requires Graphviz)

## Requirements
TAG is implemented in Python (version: 3.8.3). 
It requires the following modules: 
- graphviz (0.16) 
- IPython (7.21.0)


