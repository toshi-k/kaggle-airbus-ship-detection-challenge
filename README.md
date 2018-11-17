Kaggle Airbus Ship Detection Challenge
====

I implemented Oriented SSD for the kaggle competition ["Airbus Ship Detection Challenge"](https://www.kaggle.com/c/airbus-ship-detection).

![diagram](https://raw.githubusercontent.com/toshi-k/kaggle-airbus-ship-detection-challenge/master/img/diagram.png)

## Example
![example](https://raw.githubusercontent.com/toshi-k/kaggle-airbus-ship-detection-challenge/master/img/example.png)

## References

- Learning a Rotation Invariant Detector with Rotatable Bounding Box [arXiv](https://arxiv.org/abs/1711.09405)
- Multiscale Rotated Bounding Box-Based Deep Learning Method for Detecting Ship Targets in Remote Sensing Images [mdpi.com](https://www.mdpi.com/1424-8220/18/8/2702)

## Acknowledgement
Some scripts for encoding/decoding RLE are forked from [Paulo's kaggle-kernel](https://www.kaggle.com/paulorzp/run-length-encode-and-decode) (Apache License 2.0).

- source/01_make_coordinates/make_coordinates.py
- source/02_oriented_ssd/lib/rle.py
