Load In Process
loaded_model = pickle.load(open(modelName, 'rb'))
result = loaded_model.predict([list of features]) # Generate features using "surface_segment_data_active_preparation" method in BaselineCRF.py to convert a list of words to a list of feature sets based on those words
result = [list of char] # B for beginning of a Morpheme, E for end of a Morpheme, M for the middle of a Morpheme, S for a single character Morpheme
using this to divide the word into its composite parts will give you the morphemes 
