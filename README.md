# video_autoencoder
Video lstm auto encoder built with pytorch. https://arxiv.org/pdf/1502.04681.pdf

This project auto-encodes videos by vectorizing consecutive screens of videos using an LSTM auto-encoder.

## Training data
The training data is a collection of cow screen images sampled from some videos. Per image is sampled for every 50 frames and 6 consecutive images are used as a training sample. Since the video is 30 frames / second, one training sample is a summary of movements in 10 seconds. We tried to analyze the movements of cows by analyzing the output movement vectors of the model.
<p>
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/100.jpg" height="100" width="130" />
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/101.jpg" height="100" width="130" />
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/102.jpg" height="100" width="130" />
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/103.jpg" height="100" width="130" />
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/104.jpg" height="100" width="130" />
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/pregnant/all/pregnant/105.jpg" height="100" width="130" />
</p>

## Preprocessing
The images are vectorized using some CNNs like Resnet before input to the LSTM auto-encoder. Here, the output vector of the last full connection layer of Resnet50 is used. So every image is vectorized 

## Learning curve
The auto-encoder is well trained according to the following learning curve.
<p>
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/train_loss.png" height="250" width="500" />
</p>

## Learned vector patterns
The movement of cows in consecutive screens is vectorized by the model. The changes of colors indicate different movements of cows.
<p>
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/learned_patterns.png" height="250" width="500"/>
</p>

## t-SNE dimension reduction
Dimension reduction on the learned vectors according to different perplexity values.
<p>
  <img src="https://github.com/shuuchen/video_autoencoder/blob/master/t-SNE.png" />
</p>

