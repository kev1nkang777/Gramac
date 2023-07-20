# Gramac

## Usage
- classification
```
python3 main.py -i '<FILE_PATH>' -m <MODEL_NAME> -c
```
- detection
```
python3 main.py -i '<FILE_PATH>' -m <MODEL_NAME>
```
note : 
> "FILE_PATH" is the path of the binary file which you want to predict.
> "MODEL_NAME" is the model to be used for prediction.  e.g. rf、knn、svm、mlp .


## Description

* The main program is a malware detector.

  * **Input** : binary file
  * **Output** : probability of each class predicted by model
  * **Flow** : 
    * reverse the bin file and extract the feature
    * load the model
    * predict
## Detail work of my implement

* We first reverse the binary file to function call graph(FCG) by r2pipe.
* Next ,to extract the attribute of FCG:


the attribute we extract:
>* No. of vertices
>* No. of edges
>* No. of in degree
>* No. of out degree
>* No. of connected Component
>* No. of loops
>* No. of parallel edges

The dimension of feature is 7.

## Files


* **detection_model** : save the detection model with .joblib
* **classification_model** : save the classification model with .joblib
* **create_feature.py** : create the feature from gpickle to feature.csv
* **FCG_to_sym** : only keep the graph node which beginning with sym
* **feature** : feature of whole FCG and sym
* **main** : the detector(classifier)
* **torch_tools** : MLP tools
* **train** : about training and saving the model
* **utils** : for parsing args

## Requirements
* python3
* radare2
* python package
  * r2pipe
  * networkx
  * joblib
  * sklearn
  * argparse

## Result
### Detector
|Model|Mode|accuracy|precision|recall|F1|
|-|-|-|-|-|-|
|RF|Train|0.9681|0.9656|0.9711|0.9684|
||Validation|0.9662|0.9644|0.9686|0.9665|
|KNN|Train|0.9230|0.9985|0.8483|0.9173|
||Validation|0.9220|0.9987|0.8461|0.9161|
|SVM|Train|0.9064|0.9232|0.8038|0.8594|
||Validation|0.9061|0.9254|0.8015|0.8690|
|MLP|Train|0.9154|0.9366|0.8177|0.873|
||Validation|0.9335|0.9068|0.8164|0.8592|

### Classifier
|Model|Mode|accuracy|precision|recall|F1|
|-|-|-|-|-|-|
|RF|Train|0.9586|0.9595|0.9586|0.9585|
||Validation|0.9566|0.9577|0.9566|0.9566|
|KNN|Train| 0.9198|0.9291|0.9198|0.9178|
||Validation|0.9189|0.9285|0.9189|0.9169|
|SVM|Train|0.8785|0.8618|0.8785|0.8538|
||Validation|0.8836|0.8726|0.8836|0.8598|
|MLP|Train|0.6415|0.5499|0.6415|0.5702|
||Validation|0.8835|0.8575|0.8835|0.8695|
