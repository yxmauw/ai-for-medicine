# [AI for Medicine](https://www.coursera.org/specializations/ai-for-medicine) Specialisation @ Coursera 

* notebooks and references

### Treat each axis image sequence as a channel (e.g. R in RGB channel)
* So instead of RGB, coronal image sequence is R channel - as a way to feed into the algorithm

### Image Registration
* important step in 3D image pre-processing
* to ensure all axis image sequences are aligned to each other e.g. coronal slices are aligned wih sagittal slices
* Because sometimes patient moves/tilt between taking images of different axes

### Segmentation
* Involves defining boundaries of target in image
* points in 2D space called pixels
* points in 3D space called voxels

### Model training for segmentation task
* Use _Soft Dice Loss_ function for optimisation

## AI For Prognosis
* Survival estimation: Kaplan-Meier estimator
* Cumulative Hazard estimation: Nelson-Aalen estimator
* Evaluation of survival models
  * Harrell's C-index
  * Permissible pairs for survival data follows principle of able to compare which patient had the worse outcome based on survival period. If unable to compare in certainty e.g. censored time period, then not permissible pair

## AI For Treatment
* Evaluation of model performance
 * Shapley method can assess feature importance despite correlated features

# Citations
* [SHAP library](https://github.com/slundberg/shap) and [notebook](https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html)
* [Lifelines library](https://lifelines.readthedocs.io/en/latest/)
* [COX model](https://www.jstor.org/stable/2985181?seq=1)
* [Random survival forest](https://arxiv.org/pdf/0811.1645.pdf)
* [Harrell C-Index](https://www.ncbi.nlm.nih.gov/pubmed/7069920)

# Uesful Resources
* SNOMED CT: paid resource for searching medical terminology synonyms
* [MTSamples](https://www.mtsamples.com/): Collection of Transcribed Medical Transcription Sample Reports and Examples
