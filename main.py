
from src.TAG.TALearner import TALearner
from src.TAG.Automaton import export_ta_xml


tss_path = 'Discretization/output.txt'

xml_path = 'output/model.xml'

learner = TALearner(tss_path=tss_path, display=True)

learner.ta.show()

learner.ta.export_ta_xml(xml_path)
print(f"UPPAAL model written to {xml_path}")




