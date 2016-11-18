## Kaggle Note 2 : SVM-based  (Alaa's approach)

### Data preprocessing

* feature vectors extracted from 20s windows are used as inputs to SVM classifier
  * for each window, pass it through six Butterworth bandpass filters corresponding to 6 Berger frequency bands. the output are squared to estimate the power in six bands. here we get 16x6=96 dimensional feature vector.  Called BFB
  * use FFT to obtain the frequency spectrum. The power in each band is approximated by summing up the magnitudes of spectrum in the related band, here we also get a 16x6=96 dimensional feature vector
  * calculate the cross-channel correlation for each tow channels to measure the similarity
* one hour segments (180 consecutive 20s windows) are used for prediction