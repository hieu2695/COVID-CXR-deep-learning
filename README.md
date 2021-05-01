# DATS6501-Capstone

----------------------------
## Introduction
  
  Coronavirus disease 2019 or COVID-19 is an infectious disease caused by a newly discovered strain of coronavirus called severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This novel coronavirus was discovered in Wuhan, Hubei province, China by the end of 2019, and spreads across the world in 2020. 
  
  Since the outbreak of the disease and its severe impacts on human life worldwide, the World Health Organization (WHO) announced COVID-19 as pandemic in March 2020. According to the most recent reports from WHO, 223 countries have been facing with COVID-19, and there are approximately 141 million infected patients and more than 3 million confirmed deaths globally. 
  
 ## Objective and Project Scope

  A major step in fighting the pandemic is to detect infected individuals at an early stage and put the patients under special treatment. 
  
  Inspired by the success of computer-aided diagnosis (CADx) in medical research and the advance of deep learning in computer vision tasks, this project aims to build a CADx system using convolutional neural network architectures to screen COVID-19 patients from Chest X-ray (CXR) images. The benefits of this method are the cost efficiency of CXR imaging technique and the high performance of CNN in image classification.
  
  In this project, CXR images are collected from public sources such as Kaggle and GitHub to generate a coronavirus data including 3 classes: healthy or normal chest, viral pneumonia and COVID-19. The purpose is to develop a CNN model with high sensitivity and specificity when classifying COVID-19 images. 
  
 ## Result
 
 After experimenting many techniques and training strategies, we combine cost-sensitive learning, transfer learning, data augmentation and differential learning rates to build a ResNet34 model that gives a sensitivity of 95.47% and a specificity of 99.55%. The result is a great improvement compared to our baseline convolutional architecture whose sensitivity is only 57.36%.
 
 However, CXR imaging technique is not recommended as a replacement for real-world coronavirus detection. More empirical research and careful examinations need be conducted before bringing the method to pratical applications.

 
 ## Future Research
 
 There are several objectives to be implemented in future research to expand the project scope and mitigate existing limitations. The diversity of our coronavirus dataset can be improved by spending more time on collecting medical images or seeking assistance from coronavirus testing laboratories and hospitals. The CXR images would be resized to a higher resolution to provide clearer pictures inside the chest and help the convolutional neural network to retrieve more crucial information. An extension of this study is to include bounding box annotation to locate special patterns and abnormalities of infected chests which are useful for developing vaccines and medical treatments. Finally, there are other pretrained models such as SqueezeNet, MobileNet, EfficientNet and Inception to be tried as the backbone architecture for our CADx system. 
 
 ## Notes
 
 Helper.py contains pre-defined classes and functions which are used in the project.
 data_generation.py, data_preprocessing.py show the feature engineering.
 baseline.py contains the baseline model; baselineCV.py is our test for cross-validation method vs the normal split in to training, validation and testing.
 resnet34_freeze.py, resnet34_entire.py are the pretrained resnet34 models with and without the convolutional base frozen.
 resnet34_focal_fc.py: resnet34 model with the convolutional base frozen and only the fully connected layer is trained. The loss function is focal loss.
 resnet34_focal_full.py: resnet34 model that is trained entirely. The weights of its fully connected layer are taken from the previous trained resnet34_focal_fc model. The loss function is focal loss.
 resnet34_cs_fc.py: resnet34 model with the pretrained convolutional base frozen and only the fully connected layer is trained. The loss function is the weighted CE.
 resnet34_cs_full.py: resnet34 model that is trained entirely. The weights of its fully connected layer are taken from the previous trained resnet34_cs_fc model. The loss function is the weighted CE.
 resnet34_diff_lr.py: resnet34 model which is trained entirely using different learning rates for its layers. The initial weights of the convolutional base are the pretrained, while the initial weights of the FC layer are randomly generated.
 (Same for VGG16 and DenseNet121).
 testing_prediction.py: the code to re-check the model performance on the testing set.
 
 
 
 
 
