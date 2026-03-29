
from TAG.TALearner import TALearner
#from TAG.Automaton import export_ta_xml

tss_path = 'Discretization/output.txt'

xml_path = 'output/model.xml'



learner = TALearner(tss_path=tss_path,display=True)

learner.ta.show(title="Final Automaton", savePng=True)


learner.ta.print()





