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
