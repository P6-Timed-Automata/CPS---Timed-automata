import json, os
from TAG.TALearner import TALearner

tss_path = 'Discretization/output.txt'
xml_path = 'output/model.xml'

with open('Discretization/symbol_map.json') as f:
    symbol_map = json.load(f)

learner = TALearner(tss_path=tss_path, display=True)
learner.ta.show()

os.makedirs('output', exist_ok=True)
learner.ta.export_ta(xml_path, symbol_map=symbol_map)
print(f"UPPAAL model written to {xml_path}")