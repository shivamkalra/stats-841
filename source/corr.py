import utils as ut
import pandas as pd
import numpy as np

patient_id = 0
data = ut.load_data_for_patient(patient_id)

# this is (1302, 16, 6, 60)
specs = data['raw_spectrograms']

# extract correlation for each time-frame
def extract_correlation(i, time_index):

    spec = specs[i]
    d = pd.DataFrame(data=spec[:, :, time_index].T).corr()
    return d.as_matrix()[np.triu_indices(16, k=1)].ravel()


final_arr = np.zeros((specs.shape[0], 120, 60))
for i in range(specs.shape[0]):
    corr_mat = map(lambda x: extract_correlation(i, x), range(60))
    final_arr[i, :, :] = np.array(corr_mat).T

data['corr_channel_bands'] = final_arr

ut.save_data_for_patient(patient_id, data)
