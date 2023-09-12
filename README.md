# Plant-species-identification

-> ABSTRACT:
The determination of plant species from field observation requires substantial botanical expertise, which puts it beyond the reach of most nature enthusiasts.
Traditional plant species identification is almost impossible for the general public and challenging even for professionals that deal with botanical problems daily, such as conservationists, farmers, foresters, and landscape architects.
Even for botanists themselves, species identification is often a difficult task.
In this project, we will be able to classify 32 different species of plants on the basis of their leaves using digital image processing techniques. The images are first preprocessed and then their shape, colour and texture-based features are extracted from the processed image.
A dataset was created using the extracted features to train and test the model. The model used was Support Vector Machine Classifier and was able to classify with 90.05% accuracy.

1.INTRODUCTION:
● Plants are of central importance to natural resource conservation.
● Plant species identification provides significant information about the categorisation of plants and their characteristics.
● Manual interpretation is not precise since it involves an individual's visual perception.
● As information technology is progressing rapidly increasing, techniques like image processing, pattern recognition and so on are used for the identification of plants on basis of leaf shape description and venation which is the key concept in the identification process.
● Varying characteristics of leaves are difficult to be recorded over time.
● Hence it is necessary to create a dataset as a reference to be used for comparative analysis.

1.2 SCOPE:
Image-based methods are considered a promising approach for species identification.
A user can take a picture of a plant in the field with the built-in camera of a mobile device and analyze it with an installed recognition application to identify the species or at least receive a list of possible species if a single match is impossible.
By using a computer-aided plant identification system also non-professionals can take part in this process.

2.PROBLEM DEFINITION:
● Imagine you are planning to start your own tea-leaves business.
● You’d want to price them depending on their quality, and if you have no idea how to do that, there is no need to worry as we have the leaf classification python code in this image identification project that will help you achieve it.
● Due to their volume, prevalence, and unique characteristics, leaves are an effective means of differentiating plant species.
● They also provide a fun introduction to applying techniques that involve image-based features.
● Solution: Thus, one can build an image classifier for plant species identification by implementing image processing techniques over images of leaves.

3.LITERATURE SURVEY:
In the past decade a lot of research has been done in order to develop efficient and robust plant identification systems.

Wu et al. [5] have proposed on of the earliest plant identification system.In their scheme, they have created their own dataset named Flavia, which has been used by various other researchers as standard dataset for their work.It consists of 1907 leaf images of 32 different plant species.In their study, they extracted 5 basic geometric and 12 digital morphological features based on shape and vein structure from the leaf images. Further, principal component analysis (PCA) was used to reduce the dimensions of input vector to be fed to the probabilistic neural network (PNN) for classification. They used a three-layered PNN which achieved an average accuracy of 90.32%.


Hossain et al. [8] extracted a set of unique featured called “Leaf Width Factor (LWF)” with 9 other morphological features using the Flavia dataset. These features were then used as inputs to PNN for classification of leaf shape features. A total of 1200 leaf images were used to train the network and then PNN was tested using 10-fold cross validation, which achieved maximum accuracy of 94% at 8th fold. The average accuracy attained was 91.41%


Wang et al. [9] proposed a robust method for leaf image classification by using both global and local features. They used shape context (SC) and SIFT (Scale Invariant Feature Transform) as global and local features respectively. K-nearest neighbor (k-NN) was used to perform classification on ICL dataset which achieved an overall accuracy of 91.30%.


Authors in [10] developed a scheme which extracted 12 common digital morphological shape and vein features derived from 5 basic features. They implemented both k-NN and support vector machine (SVM) which attained an accuracy of 78% and 94.5% respectively when tested on Flavia dataset


Pham et al. [11] in their computer-aided plant identification system compared the performance of two feature descriptors i.e. histogram of oriented gradients (HOG) and Hu moments. For classification, they selected SVM due to its ability to work with high dimensional data.They obtained accuracy of 25.3% for Hu moments and 84.68% for HOG when tested with 32 species of Flavia dataset


Mouine et al. [12], in their study introduced new multiscale shape-based approach for leaf image classification. They studied four multiscale triangular shape descriptors viz. Triangle area representation (TAR),Triangle side length representation (TSL), Triangle oriented angles (TOA) and Triangle side lengths and angle representation (TSLA). They tested their system on four image datasets: Swedish, Flavia, ImageCLEF 2011 and ImageCLEF 2012. With Swedish dataset they computed classification rate as 96.53%, 95.73%, 95.20% and 90.4% for TSLA, TSL, TOA and TAR respectively using 1-NN.


Authors in [13] proposed a method for plant identification using Intersecting Cortical Model (ICM) and used SVM as the classifier. This study used both shape and texture features viz. Entropy Sequence (EnS) and Centre Distance Sequence (CDS). They attained accuracy of 97.82% with Flavia dataset, 95.87% with ICL1 and 94.21% with ICL2 (where ICL1 and ICL2 are subsets of ICL dataset)

4.PROJECT DESCRIPTION:
The dataset used is Flavia leaves dataset which also has the breakpoints and the names mentioned for the leaves dataset
● contains create_dataset() function which performs image pre-processing and feature extraction on the dataset. The dataset is stored in Flavia_features.csv
● uses extracted features as inputs to the model and classifies them using SVM classifier
● contains exploration of preprocessing and feature extraction techniques by operating on a single image

DATASET USED:
Flavia data set: This data set contains 1907 leaf images of 32 different species and 50–77 images per species. Those leaves were sampled on the campus of the Nanjing University and the Sun Yat-Sen arboretum, Nanking, China. Most of them are common plants of the Yangtze Delta, China (Wu et al., 2007). The leaf images were acquired by scanners or digital cameras on a plain background. The isolated leaf images contain blades only, without petioles.


4.1.PROPOSED DESIGN:

<img width="568" alt="Screenshot 2023-09-12 at 10 06 00 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/be855807-bdd8-4d09-976a-e792b81b8b05">

5.REQUIREMENTS:
● PC
● FLAVIA DATASET
● JUPYTER NOTEBOOK
● ANACONDA DISTRIBUTION

5.1.FUNCTIONAL REQUIREMENTS:
LIBRARIES USED:
● Numpy
● Pandas
● OpenCV
● Matplotlib
● Scikit Learn
● Mahotas

6.METHODOLOGY:

I. Pre-processing:
The following steps were followed for pre-processing the image:
1. Conversion of RGB to Grayscale image
2. Smoothing image using Guassian filter
3. Adaptive image thresholding using Otsu's thresholding method
4. Closing of holes using Morphological Transformation
5. Boundary extraction using contours

II.Feature extraction:
Variou types of leaf features were extracted from the pre-processed image which are listed as follows:
1. Shape based features : physiological length,physological width, area, perimeter, aspect ratio, rectangularity, circularity
2. Color based features : mean and standard deviations of R,G and B channels
3. Texture based features : contrast, correlation, inverse difference moments, entropy

III.Model building and testing:
(a) Support Vector Machine Classifier was used as the model to classify the plant species
(b) Features were then scaled using StandardScaler
(c) Also parameter tuning was done to find the appropriate hyperparameters of the model using GridSearchCV

7.EXPERIMENTATION:
PRE-PROCESSING
Converting image to grayscale:
First, color images were converted into grayscale images

<img width="300" alt="Screenshot 2023-09-12 at 10 18 20 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/0d201c3f-0619-4916-bb67-2108d05c9d05">

Smoothing image using Guassian filter of size (25,25):
Noise handling is an important task in image processing.Therefore, the Gaussian filter was used to reduce noises of the grayscale image as shown in Figure 01, which is also called as Gaussian smoothing.


<img width="600" alt="Screenshot 2023-09-12 at 10 19 49 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/85ec0b6f-0a0b-470e-a42e-3f4d44eb7364">

Adaptive image thresholding using Otsu's thresholding method:
filter. 
Then, the enhanced images were transformed into binary images using the Otsu threshold method that is usually used to separate pixels into two classes. These binary images may contain imperfections. Therefore, morphology operation was used to reduce them, which was employed on binary images as shown in Figure 04.


<img width="300" alt="Screenshot 2023-09-12 at 10 23 48 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/f0256824-2bcf-456c-99dd-f247d8b86c31">



Closing of holes using Morphological Transformation


<img width="300" alt="Screenshot 2023-09-12 at 10 25 10 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/3d2de9f8-0e9f-4cfc-8136-b19951a2bcbe">


Boundary extraction
Boundary extraction is needed which will be used in calculation of shape features.


Boundary extraction using sobel filters - Not effective


<img width="300" alt="Screenshot 2023-09-12 at 10 26 15 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/7eb22a5b-7ba9-4f83-a972-01120b4c84cd">


Boundary extraction using contours - Effective


<img width="300" alt="Screenshot 2023-09-12 at 10 28 03 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/d1302ab6-a2e5-4c65-af0d-f5e63bb86312">

FEATURE EXTRACTION:
This study used three types of features of a leaf which are shape, texture, and color features.

Calculating moments using contours
First, the study extracted area and perimeter features by calculating moments using contours. Our approach was developed on fully-grown and not tempered leaf images as the study conducted by Kaur Kaur (2019).

Fitting in the best-fit rectangle and ellipse:
Then our approach focused on generating the best-fit rectangle and ellipse as shown in Figure below in order to extract another three shape features for the feature space which are aspect ratio, rectangularity, and circularity. Apart from these features, length and width were also extracted as shape features.

<img width="600" alt="Screenshot 2023-09-12 at 10 31 13 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/43937df0-f512-44a4-96c5-a8711c2e432a">

Texture-based features are also an important feature category of a leaf. The term texture defines various properties of an image like coarseness, smoothness, and regularity in image processing. Four texture features such as contrast,correlation, inverse difference moments, and entropy were extracted in this study using GLCM.

Calculating color based features - mean, std-dev of the RGB channels
Color-based features can be used in image classification.Each digital color image pixel is a combination of RGB (Red, Green, Blue) values. Thus,the RGB color model of a digital image consists of three components as shown in Figure below

<img width="600" alt="Screenshot 2023-09-12 at 10 33 50 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/7f1a0f25-e230-4e2a-81bb-6ea2038314bf">



8.TESTING AND ANALYSIS:
Classification in our work, typically means to assign a certain plant species to the image based on the feature set extracted. In other words, classification is a process of identifying the class label of a new input image on the basis of the prior knowledge (training dataset). For our study, we have used a supervised classification technique in which the labels of the classes (here, plant species) are already known and the new data input is assigned to one of the labels.

SVM CLASSIFIER:
In machine learning, support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data and recognize patterns, used for classification and regression analysis. Given a set of training examples, each marked for belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples into one category or the other, making it a nonprobabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible.

<img width="450" alt="Screenshot 2023-09-12 at 10 36 28 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/a62ceb4c-4be4-4c44-9286-40968686fbae">

ACCURACY:
0.90052356020942403

9.OUTPUT:

<img width="430" alt="Screenshot 2023-09-12 at 10 38 27 PM" src="https://github.com/keerthanamg/Plant-species-identification/assets/88154987/d3f83645-ea45-40fa-8678-b6a18960eac9">


9.REFERENCES:

[1] D. Bambil et al., "Plant species identification using color learning resources, shape, texture, through machine learning and artificial neural networks," Environment Systems and Decisions, vol. 40, no. 4, pp. 480-484, 2020, doi: 10.1007/s10669-020-09769-w.

[2] H. U. Rehman, S. Anwar and M. Tufail, "Machine vision-based plant disease classification through leaf imagining," Ingenierie Des Systemes d'Information, vol. 25, no. 4, pp. 437-444, 2020, doi: 10.18280/isi.250405.

[3] A. Soleimanipour and G. R. Chegini, "A vision-based hybrid approach for identification of anthurium flower cultivars," Computers and Electronics in Agriculture, vol. 174, p. 105460, 2020, doi: 10.1016/j.compag.2020.105460.

[4] R. A. Asmara, M. Mentari, N. S. Herawati Putri and A. Nur Handayani, "Identification of Toga Plants Based on Leaf Image Using the Invariant Moment and Edge Detection Features," 2020 4th International Conference on Vocational Education and Training (ICOVET), 2020, pp. 75-80, doi: 10.1109/ICOVET50258.2020.9230343.

[5] I. Ariawan, Y. Herdiyeni and I. Z. Siregar, "Geometric morphometric analysis of leaf venation in four shorea species for identification using digital image processing," Biodiversitas, vol. 21, no. 7, pp. 3303-3309, 2020, doi: 10.13057/biodiv/d210754.

[6] M. Y. Braginskii and D. V. Tarakanov, "Identification of plants condition using digital images," Journal of Physics: Conference Series, vol. 1353, no. 1, pp. 1-4, 2019, doi: 10.1088/1742-6596/1353/1/012108.

[7] Y. Ampatzidis et al., "Vision-based system for detecting grapevine yellow diseases using artificial intelligence," Acta Horticulturae, vol. 1279, pp. 225-230, 2020, doi: 10.17660/ActaHortic.2020.1279.33.

[8] G. Ramesh, D. W. Albert and G. Ramu, "Detection of plant diseases by analyzing the texture of leaf using ANN classifier," International Journal of Advanced Science and Technology, vol. 29, no. 8, pp.1656-1664, 2020

[9].G. Singh, N. Aggarwal, K. Gupta and D. K. Misra, "Plant Identification Using Leaf Specimen," 2020 11th International Conference on Computing, Communication and Networking Technologies (ICCCNT), 2020, pp. 1-7, doi: 10.1109/ICCCNT49239.2020.9225683.

[10].G. Cerutti, L. Tougne, J. Mille, A. Vacavant and D. Coquin, "A model-based approach for compound leaves understanding and identification," 2013 IEEE International Conference on Image Processing, 2013, pp. 1471-1475, doi: 10.1109/ICIP.2013.6738302.

[11].D. C. Nadine Jaan et al., "A Leaf Recognition of Vegetables Using Matlab," International Journal of Scientific & Technology Research, vol. 5, no. 2, pp. 38-45, 2016.

[12].Y. Li, J. Nie and X. Chao, "Do we really need deep CNN for plant diseases identification?," Computers and Electronics in Agriculture, vol. 178, p. 105803, 2020, doi: 10.1016/j.compag.2020.105803.






