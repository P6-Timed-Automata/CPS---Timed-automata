from src.TAG.TALearner import TALearner
#tss = ["a:2 b:5", "a:3 c:6"]


# merge example
#tss = ["a:2 b:5", "c:1 a:3 b:4"]


#tss = ["a:2 b:4", "a:3 b:5", "a:7 c:9", "a:8 c:10"]

tss = ["c:4 m:4 h:3 m:5 c:8 m:6"]



#tss = ["a:1 b:1 c:3", "a:2 b:1 c:1", "a:1 c:2"]


learner = TALearner(tss_list=tss, display=True)

learner.ta.show()





