Acc: Walles
PW: Zoop$012


# Introduction
The codebase operates on a root directory with an expected structure, and then processes all images\
(and associated detailing .csv files) in one go.

The input is a series of folders of the expected structure.\
The output is many-fold. <<TBD>>

# How to Run

The codebase is to be executed in the following order:

1. Remove duplicate images through image hashing: `python3 duplicate_image_removal.py -d ./profiles -r 1`
2. Renaming profiles to depth profiles: `python3 modified_renaming_profiles.py`
3. Flatfielding profiles: `python3 flatfielding_profiles.py`
4. Object detection from flatfielded profiles: `python3 object_detection.py`
5. Classification of detected objects: `python3 object_classification.py`
6. Processing of classification: `python3 class_viewer.py`