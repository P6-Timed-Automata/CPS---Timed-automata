from TAG.TALearner import TALearner
import os

tss_path = 'Discretization/output.txt'
xml_path = 'output/model.xml'

learner = TALearner(tss_path=tss_path, display=True)
learner.ta.show()

os.makedirs('output', exist_ok=True)
learner.ta.export_ta(xml_path)
print(f"UPPAAL model written to {xml_path}")