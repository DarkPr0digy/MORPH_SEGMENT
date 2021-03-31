Unsupervised morpheme segmentation in Nguni languages using bi-entropy models
Morfessor-Baseline included as baseline measure

To use, 
1. Upload colab_data to 'My Drive/Colab Notebooks' in google drive
2. Upload Unsupervised_morpheme_segmentation.ipynb to 'My Drive/Colab Notebooks' in google drive
3. Open and run each cell in Unsupervised_morpheme_segmentation.ipynb, after giving google colab access to google drive 

colab_data contains 2 folders: entropy_data and morfessor_data.
Each folder contains all data for the respective models.

Training data set for morfessor baseline is in *.training.txt, while testing dataset is in *.test.txt, where * is one of [ndebele, xhosa, swati, zulu].
Trained models are stored in saved_models folder for later use. 

colab_data contains 2 folders: Language model 1 (LM1), Language model 2 (LM2), and *_data, where * is one of [ndebele, xhosa, swati, zulu].
*_data contains folders right and left. 
Both folders right and left contain data sets for right and left entropy, respectively, for the training of LM1.
Both folders right and left contain data sets for training, validation, and testing; 
train.txt in *_data is input for LM2 training
*_data contains the data set to be evaluated in evaluate.txt

Both folders LM1_models and LM2_models contain pretrained left and right entropy models, for left and right entropy respectively, for each language.
 