## Kaggle Note 1 : CNN (10th approach)

### Model evaluation

* seizure activity is highly individual, models are mostly tuned for each patient separately.
* Combination of AUC is tricky. calibration of predictions is important. Here we use unity-based normalization 

$$
p_s = \frac{p - min(p)}{max(p)-min(p)}
$$

* some other tricks 
  * scale prediction by a logistic function to (0, 1), which is "softmax scaling"
  * for each subject, take the median of predictions, subtract it from scores and then divide the scores by 2 and add 0.5, which is "median scaling"

### Data preprocessing

Previous studies showed that features from the frequency domain are effective for seizure prediction(Howbert et al., 2014)



it is important to consider the relationships between pairs of EEG channels (Mirowski et al., 2009)

* FFT

  * convert the signal in a time domain to the frequency domain

* resample the signal to 400Hz and apply a band-pass filter between 0.1~180 Hz.

  * Delta wave 0.1~4 Hz
  * Theta wave 4~8 Hz
  * Alpha wave 8~12 Hz
  * Beta wave 12~30 Hz
  * Lowgamma 30 ~70 Hz
  * Highgamma 70~180 Hz

* Each 10 minutes time series was partitioned into non-overlapping 1 min windows, for each frame we calculated log(10) of ites amplitude spectrum, In each frequency band we took an average of the corresponding log(Ak), the dimension of the data clip is channels x frequency bands x windows, 16x6x10

* Standardized the spectrograms from each channel

  * Before being used in CNN, flat the spectrogram into a vector of 60 values, from each value substact the mean and divide by standard deviation of complete train dataset. (By doing so, we account for the position of a feature both in time and frequency)
  * decompose each 6x10 spectrogram into 10 examples each of 6 features, standardize these 6-dimensional vectors on per feature basis and combine them back into 6x10 matrix. (Here we do not account for the time location)

* Data augmentation

  * take overlaps between consecutive 10 minute clips, this can be done during training, here we do not need to store additional data

  ​

  ​

 