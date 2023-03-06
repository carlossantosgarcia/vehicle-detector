# Vehicle Detector
This repository provides an implementation for a vehicle detection algorithm based on classical computer vision features (HOG, SIFT) and machine learning classifiers like support vector machines or gradient boosting classifiers. To install the required packages, run:
```
pip install -r requirements.txt
```
This code relies on the dataset shared for the 2023 Visual Computing Kaggle Challenge. ```train``` and ```test``` folders are supposed to be nested in the ```data``` folder. 

# Report
The report for this project is available [here](report.pdf)

# Reproducing Kaggle submission
To reproduce my final submission, run:
```
python3 inference.py --test_dir data/test --clf_path models/gradient_boosting.pkl --scaler_path models/scaler.pkl --submission
```
This relies on a Gradient Boosting classifier trained on 64x64 patches extracted from the training dataset, and applied in a sliding window manner on test images. 

# Training a classifier
To train a classifier, run:
```
python3 train.py --model gradient_boosting --bow --spatial --hist
```
Possible models are ```linear_svm```, ```svm``` or ```gradient_boosting```.

# Project organization
- ```bag_of_words.py```: Creates the vocabulary for Bag-of-SIFT features.
- ```dataset.py```: Defines a class that simplifies the creation balanced dataset of patches either containing (postive samples) or not containing (negative samples) any vehicles and the extraction of features.
- ```inference.py```: Defines the inference pipeline that applies the trained classifiers on a sliding window manner on test images.
- ```train.py```: Launches training runs of models