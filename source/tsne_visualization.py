import utils as iu
import numpy as np
import seaborn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


patient_id = 1
data = iu.load_data_for_patient(patient_id)

ch_idx = 3


channel_id = 3
freq_level = 1
X = data['raw_spectrograms'][:, channel_id, freq_level, :]


print "Done"
model = TSNE(n_components=2, verbose=1)
Y = model.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1], c=data['target'])
plt.show()
