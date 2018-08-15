# Machine Learning Using AutoKeras

## Installation

One way to install all the required dependencies, is to use virtual environment. The steps are as follows: <br>
* pip install virtualenv
* in bash, run the following command: "venv\Scripts\activate.bat"
* in bash, run the following command: "pip install -r requirements.txt", enjoy

## Description
Use of an open source library [AutoKeras](https://autokeras.com/) for automated machine learning.
AutoKeras provides the functions to automatically search for architecture
and parameters for deep learning models. Different datasets were used for evaluation of library

## Pre-processing strategy
In the cause of Olivetti Faces dataset, because data was sampled from 40 individuals and each class only had 10 samples, the size of the dataset was increased by 4x by artificially synthesizing data, this was achived by rotating each image 90, 180 and 270 degrees counter clockwise <br>

## Results

| Dataset | No. of Samples | Data Augmentation | Train Accuracy (%) | Test Accuracy (%) | Average Precision (%) | Average Recall (%) | Average F1 Score (%) | Total Support | Epochs | Time Taken (s) | Batch Size |
| :-: | :-: | :-: | :-: | :-: | :-:| :-: | :-: | :-: | :-: | :-: | :-: |
| [Olivetti Faces Dataset](http://scikit-learn.org/stable/datasets/olivetti_faces.html) | Yes | 1600 | 94.393 | 95.938 | 96 | 97 | 96 | 320 | 46 | 96.859 | 128 |
| [Cifar-10 Dataset](https://en.wikipedia.org/wiki/CIFAR-10) | 60000 | No | 69.278 | 69.180 | 69 | 70 | 69 | 10000 | 72 | 1876.29 | 128 |
| [Cifar-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (un-normalized features) | 60000 | No | 40.1440 | 39.66 | 39 | 41 | 40 | 10000 | 54 | 1370.663991 | 128
| [Mnist](https://en.wikipedia.org/wiki/MNIST_database) | 60000 | No | 98.492 | 98.350 | 98 | 98 | 98 | 10000 | 37 | 98.350 | 128 |

The classification report containing average precision, recall, f1-score and total support along with the precision, recall, f1-score and support for each class is given in
the file 'classification_report.csv'. <br>

## Limitations
* It seems like AutoKeras does not support architectures such as ResNet.
* Cifar-100 suffers from very low test and train performance as compared to the baseline model
* There is limited support in Keras for things like plotting the 'ROC Curve' with multiple classes
* Things such as transfer learning become more difficult to achieve

## Issues
At this point in time, there are some issues with the Windows version of
Auto-Keras given that you use 'CUDA' device instead of 'CPU'. It was 
experienced that the multiprocessing API and PyTorch do not work well together. The issues
are listed as follows: <br>

* "Traceback (most recent call last):
<br>... <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
TypeError: 'float' object cannot be interpreted as an integer". <br>
 Don't be mislead into thinking that the problem is related
to the multiprocessing API, it simply refers to explicitly casting values in the tuple
'self.padding' as int (C:\Users\user\AppData\Local\Programs\Python\Python36\site-packages\torch\nn\modules\conv.py at 
line 301 before calling the function F.conv2d with appropriate parameters)

* After fixing the former error, when you run the 'auto_keras_magic.py' script again, the program will fail with the following information
"Traceback (most recent call last):
<br>... <br>
THCudaCheck FAIL file=c:\new-builder_3\win-wheel\pytorch\torch\csrc\generic\StorageSharing.cpp line 231 error 71: operation not supported 
<br> ... <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\lib\site-packages\autokeras\search.py", line 190 in search <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
multiprocessing.pool.MaybeEncodingError: error sending result '[(98.08, tensor=(2.3784, device='cuda:0'), <autokeras.graph.Graph.object at 0x000002821B58E668>)]' <br>
Reason: 'RuntimeError('cuda runtime error (71) : operation not supported at c:\\new-builder_3\\win-wheel\\pytorch\\torch\\csrc\\generic\\StorageSharing.cpp:231',)' <br>
Unfortunately it looks like multiprocessing and PyTorch do not seem to work well together. A hack to this problem is to replace line 178: <br>
"train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args, <br>
                                                os.path.join(self.path, str(model_id) + '.png'), self.verbose)])" <br>
with: <br>
train_results = train((graph, train_data, test_data, self.trainer_args, os.path.join(self.path, str(model_id) + '.png'), self.verbose)) <br>
and replace line 190 <br>
accuracy, loss, graph = train_results.get()[0] <br>
with: <br>
accuracy, loss, graph = train_results <br>
