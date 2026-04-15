import numpy as np
from scipy.io import arff

from Discretization import Persist


# Breakpoints creation
#f_train = "Chinatown_TRAIN.arff"
#df_train, meta = arff.loadarff(f_train)

trace = "../Data/3-ExtractInterval/A/1day/A-1day-tid1.csv"

#ts = np.array(df_train[list(meta._attributes.keys())[:-1]].tolist())
p = Persist(x = trace, divergence="w", candidates="EW")


bkpts = [b for b in p.bins[np.argmax(p.pscores)] if not np.isnan(b)] # best breakpoints

# Vizualize breakpoints on an instance of time series
import matplotlib.pyplot as plt

plt.plot(ts[0])
for b in bkpts:
    plt.axhline(b, color="red")
