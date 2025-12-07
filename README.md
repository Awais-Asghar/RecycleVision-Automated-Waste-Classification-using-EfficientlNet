# RecycleVision: Automated Waste Classification using EfficientNet
![Project Status](https://img.shields.io/badge/Project_Status-Completed-brightgreen.svg)
![Model](https://img.shields.io/badge/Model-EfficientNetB0-blue.svg)
![Environment](https://img.shields.io/badge/Environment-Kaggle-orange.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_|_Keras-yellow.svg)
![Method](https://img.shields.io/badge/Method-Transfer_Learning-red.svg)
![DeepLearning](https://img.shields.io/badge/Deep_Learning-Enabled-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Manual waste sorting is slow, boring, and error prone. Recycling bins are often filled with mixed waste, so a lot of potentially recyclable material ends up in landfills. RecycleVision is a deep learning based image classifier that can automatically identify different types of waste from images. Using transfer learning with EfficientNetB0, the system learns rich visual features and classifies images into multiple waste categories. This model is the “brain” that can later be deployed on an embedded device or robotic arm to perform automatic, real-time waste segregation, making recycling cheaper, faster, and more reliable. 

---

## 1. Problem Statement & Motivation

**Core Problem**
Given an image of a waste item, predict its correct category (cardboard, glass, metal, paper, plastic, or trash). This helps automate waste segregation in smart bins and recycling plants. 

**Why this problem is important**

* Incorrect segregation sends recyclable material to landfills.
* Manual sorting is labor intensive, slow, and unhygienic.
* An automated vision system can continuously sort waste without fatigue.
* Such a model can be integrated into smart cities, recycling centers, and robotic sorting lines.

**Project Goal**

Design, implement, and evaluate a deep learning model using EfficientNetB0 and Keras/TensorFlow to classify waste images with high accuracy, while keeping the solution lightweight enough for future deployment on resource-constrained hardware. 

---

## 2. Dataset Description

**Source**

* Kaggle: Garbage Classification / Garbage Classification v2 dataset. 

**Classes**

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

**Data Characteristics**

* Color images with different backgrounds, orientations, and lighting conditions.
* Imbalanced class distribution (some classes have more images than others), visualized in the bar chart on page 8 of the report. 

**Preprocessing Steps**

* Image resizing to 224 × 224 pixels.
* Rescaling pixel values to [0, 1].
* Splitting into train, validation, and test sets using stratified train_test_split. 

**Data Augmentation**

To increase robustness and reduce overfitting:

* Random rotations
* Width and height shifts
* Shear and zoom transformations
* Horizontal flips
* Nearest fill mode

This is implemented using `ImageDataGenerator` for training, with only rescaling for validation and test sets. 

---

## 3. Model Architecture (EfficientNetB0 Transfer Learning)

**High-Level Idea**

Use a pre-trained CNN (EfficientNetB0) as a fixed feature extractor and add a small custom classification head on top. This is faster and needs less data compared to training from scratch. 

**Base Network**

* EfficientNetB0 pre-trained on ImageNet.
* `include_top=False` to remove the original classification head.
* Input shape: 224 × 224 × 3.
* Base model is frozen initially to preserve learned features.
* Loaded weights file shown in the screenshot on page 10 (efficientnetb0_notop.h5). 

**Custom Classification Head**

1. Global Average Pooling 2D
2. Batch Normalization (as shown in model summary on page 10) 
3. Dense layer(s) with ReLU activation
4. Dropout for regularization
5. Final Dense layer with 6 units and Softmax activation for multi-class output

**Training Setup**

* Loss: Categorical Crossentropy
* Optimizer: Adam with a small learning rate (~1e-4)
* Metrics: Accuracy, Precision, Recall, F1-Score
* Training for multiple epochs with early stopping based on validation performance. 

---

## 4. Training & Evaluation

**Data Split**

* Train set
* Validation set
* Test set

All splits preserve class ratios using stratified sampling. 

**Key Training Observations**

* Training and validation accuracy curves steadily increase and converge, while loss curves decrease, indicating good learning without severe overfitting (plots on page 10). 

**Final Test Results**  (from page 11) 

* Test Accuracy ≈ **92.9%**
* Test loss ≈ 0.22
* Precision ≈ 0.95
* Recall ≈ 0.92
* Macro F1-Score ≈ 0.93

**Confusion Matrix Insights**

* Most classes are classified correctly with high support.
* Main confusion happens between visually similar materials such as clear plastic vs glass.
* The confusion matrix on page 11 shows strong diagonal dominance, which confirms overall good performance. 

---

## 5. System Pipeline (For a Slide Diagram)

1. **Image input**

   * Capture or upload a waste item image.

2. **Preprocessing**

   * Resize to 224 × 224.
   * Normalize pixel values.

3. **EfficientNetB0 Feature Extraction**

   * Pass image through frozen EfficientNetB0 backbone.

4. **Custom Classification Head**

   * Apply global average pooling, dense layers, dropout, and softmax.

5. **Prediction**

   * Output a probability distribution over 6 classes.
   * Choose the class with highest probability.

6. **Application Layer**

   * Use prediction to decide which bin / conveyor belt to send the item to (future hardware integration). 

---

## 5. Discussion, Limitations, and Future Work

**What worked well**

* Transfer learning using EfficientNetB0 gave high accuracy with relatively low training time.
* Data augmentation improved generalization to unseen orientations and lighting.
* The model is compact enough to be considered for deployment on edge devices. 

**Limitations**

* Dataset images are relatively clean and centered; real-world waste is messier.
* Confusion remains for visually similar materials.
* The model handles single objects in an image, not multiple items at once. 

**Future Work**

* Collect more realistic data: dirty, crushed, partially visible items.
* Fine-tune upper layers of EfficientNetB0 on this dataset.
* Use object detection models (for example YOLO) to handle multiple waste items per frame.
* Integrate with hardware (embedded controller / robotic arm) for real-time sorting in a lab prototype. 

---

## 6. Key Technical Terms (Quick Glossary)

* **Transfer Learning**
  Using a model pre-trained on a large dataset (ImageNet) and adapting it to a new but related problem.

* **EfficientNetB0**
  A family member of EfficientNet models that scale width, depth, and resolution in a balanced way to give high accuracy with fewer parameters.

* **Global Average Pooling**
  Replaces large fully connected layers by averaging spatial features, reducing parameters and overfitting.

* **Softmax**
  Converts logits into a probability distribution over all classes.

* **Confusion Matrix**
  A table that shows correct and incorrect predictions for each class, helping analyze model behavior.

* **Macro F1-Score**
  Average F1-Score over all classes, giving equal importance to each class even when data is imbalanced.

---
