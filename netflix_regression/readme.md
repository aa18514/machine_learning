# ridge regression - netflix dataset

## Running the application 

use verbosity (-v/-vv/-vvv) to switch between naive linear regression, regression with L2 regularization and regression with transformed features with regularization <br> 

## Dataset 

Dataset consits of 671 users and 9066 movies <br> 
All the data exists in the subdirectory "\movie-data" <br>

## Input features

The input features originally have a dimension of d + 1 where d = 18 and in this case x[0] = 1 <br> 
x[0] is used for the constant offset term, the rest of the terms represent different movie genres <br> 

## Output 

The predictor function, y_hat represents the expected rating <br>
The target function, y represents the actual rating <br> 

## Stratagies

### Prepocessing the input features

All feature vectors are normalized to zero and unit variance, the same parameters are used to normalize the test data set. <br>
The step is done before the data is partitioned according to different users, where for each user we derive an optimal weight vector <br>
In the case of non-transformed features the size of the feature vector is 19, and in the case of transformed features the size of <br>
the feature vector is 172 <br>

### Controlling Overfitting

L2 regularization is used to reduce overfitting (https://en.wikipedia.org/wiki/Overfitting) and improve test accuracy <br> 
We learn the regularized weights for each user seperately which leads to higher bias and lower bias as compared to taking a single 
regularized weight vector for all the users <br> 

Values of lambda were taken to be 10**x where a 100 samples of x are taken between -5 and 0, by invoking the function np.logspace(-5, 0, 100). 

The following graph shows the lambda values for all 671 users: 
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/lambda_values.png" /> 
</p> 
the loss function in this case is taken to be L2 norm (Euclidean length between the predictor and the target) <br> 
Added support for the multi-processing module to parallelize against different values of K respectively <br> 

### Speeding up compute

The joblib libarray was used to distribute work amongst multiple cores - in this case the process of finding an optimal weight for each user, and getting the train and test bias 
and variance for the user. This has been tested on 4 cores, but in the future the plan is to move the compute to **Amazon AWS** with support for upto 32 cores (in which case simply 
distributing work amongst multiple cores won't work and there needs to be some sort of agglomeration of events. 

I am also tempted to move away from NumPy to TensorFlow, identify potential hotspots and move them to Amazon GPU/FPGA instances. I am also tempted to play around with different data types 
and their affect on the accuracy. 


### Applying non-linear transformation

The non-linear transformation is taken to be each genre multiplied with the rest of the genres in the dataset. <br>
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/correlation_coefficients.png" />
</p>
The following figure shows the correlation coefficients between different genres in the dataset; it is worth noting <br> 
that the correlation coefficient between the genre 'Comedy' and 'Drama' is -0.61976, which shows that most of the movies <br>
that contain the genre 'Comedy' do not contain the genre 'Drama', and vice-versa  <br> 
the original features remain unchanged, the transformed features are appended to the original feature vector <br>
This yields a new dimension vector with a dimension of 172 <br>  

## Results

The following figures show the exponentially weighted training and test errors for 671 netflix users, which makes it more conveniant for us to capture the trends in the training and test bias. 
The exponentially weighted average is calculated as follows: <br>
<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/for.gif" width="242" height="19" /> <br>
<p align="center"> 
<img src="http://www.sciweavers.org/tex2img.php?eq=%20%20v_%7Bt%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="  v_{t} " width="21" height="15" /> 
= 
<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/CodeCogsEqn.gif" align="center" border="0" alt=" \beta " />
* 
<img src="http://www.sciweavers.org/tex2img.php?eq=%20%20v_%7Bt%20-%201%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="  v_{t - 1} " width="39" height="15" /> 
+ (1
- 
<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/CodeCogsEqn.gif" align="center" border="0" alt=" \beta " />)
*
<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/thetha.gif" align="center" border="0" alt=" \beta " /> 
 </p> 
where v(0) is initialized to zero <br> 
after each iteration the value if meanPrev is updated to the value of meanNext respectively <br>
the value of beta chosen for the analysis is 0.9, although in the future this can be experimented choosing an appropiate value of beta can be experimented with in the future. <br>
Taking the value of beta equal to 0.9 is analagous to taking the mean over the last 10 iterations <br>
<br>
<div>
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/bias_against_K.png" width="400" height="400" />
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/variance_against_K.png" width="400" height = "400" /> 
</div>

<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/total_error.png" width="400" height="400" /> 
</p>
<div>
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/test_train_error.png" width="400" height="400" />
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/test_train_regularized.png" width="400" height = "400" /> 
</div>
<br>
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/non-linear-features.png" width="400" height="400" /> 
</p>

|  | Mean Test Bias | Mean Train Bias | Mean Test Variance | Mean Train Variance | time (s) |
| :---: | :-: | :-: | :-:| :-: | :-: |
| **Unregularized (Original Features)**  | 1.932715 | 0.564595 | 2.416220 | 0.130706  | 006.669397 |
| **Regularized (Original Features)**    | 1.889901 | 0.572407 | 2.276397 | 0.129983  | 476.689539 |
| **Regularized (Transformed Features)** | 3.869358 | 0.297641 | 5.153465 | 0.068069  | 2100.70408 |

While for the original features, we see that moving from unregluarized version to L2 - regularization has resulted in a decline in the test bias, for the transformed features, the test bias nearly doubles <br> 
This suggests that we have too many features, and limited dataset. This emphasizes the need for feature reduction techniques such as principal components. <br> 
After the training and test data was using non - linear transformation and standardized, PCA was used to project the features into a new dimension space. The test and train bias were recorded with respect to <br> 
the number of principal components: <br> 
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/error1.png" width="400" height="400" /> 
</p>
The following diagram shows the train and the test variances with respect to the number of principal components: <br> 
<p align="center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/variance.png" width="400" height="400" />
</p> 
Similarly, I tried to capture the computational complexity of principal component analysis with respect to the number of features: <br> 
<p align = "center"> 
	<img src="https://github.com/aa18514/Python/blob/master/netflix_regression/images/times.png" width="400" height="400" /> 
</p>          