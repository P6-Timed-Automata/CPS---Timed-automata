from src.TAG.TALearner import TALearner

tss_path = 'Discretization/output.txt'



learner = TALearner(tss_path=tss_path, display=True)

learner.ta.show()





