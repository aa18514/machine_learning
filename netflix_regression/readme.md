# ridge regression - netflix dataset

## Running the application

Use verbosity (-v/-vv/-vvv) to switch between naive linear regression, regression with L2 regularization and regression with non-linear transformation of the features with regularization

## Dataset

Dataset consists of 671 users and 9066 movies <br>
All the data exists in the subdirectory "\movie-data" <br>

## Input features

The input features have a dimension of d + 1 where d = 18  and in this case x[0] = 1, which corresponds to the bias term <br>. The rest of the terms represent different movie genres <br>


## Output

The predicted value y<sup>~</sup> represents the expected rating <br>
The target function, y represents the actual rating <br>

## Strategies

### Preprocessing the input features

All feature vectors are normalized to zero mean and unit variance, the same parameters are used to normalize the test data set. <br>
The step is done before the data is partitioned according to different users; for each user we derive an optimal weight vector. <br>
In the case of non-transformed features the size of the feature vector is 19, and in the case of transformed features the size of the feature vector is 172. <br>

### Controlling Overfitting

L2 regularization is used to reduce [overfitting](https://en.wikipedia.org/wiki/Overfitting) and improve test accuracy. <br> 
A high value of &#955; you will risk [underfitting](https://en.wikipedia.org/wiki/Overfitting#Underfitting), whereas a low value of &#955; you will risk [overfitting](https://en.wikipedia.org/wiki/Overfitting). We choose the value of &#955; by using [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). <br>
We learn the regularized values for each user seperately higher variance and lower bias as compared to taking as compared to a single regularized weight vector for all the users <br>
Values of lambda were taken to be 10<sup>x</sup> where samples of x are taken between - 5 and 0, by calling the command: 
```python
np.logspace(-5, 0, 100)
```

The following graph shows the lambda values for all 671 users:
<p align="center">
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/lambda_values.png">
</p>

The loss function in this case is taken to be the [L2 norm](https://mathworld.wolfram.com/L2-Norm.html) <br>
Added support for the multiprocessing module to parallelize against different values of K respectively <br>

### Accelerating Compute

The joblib library was used to distribute work amongst multiple cores - in this case the process of finding an optimal weight vector for every user and getting the train and test bias and train and test variance for each user. This has been supported on 7 cores, but in the future the plan is to move the compute to [Amazon AWS](https://aws.amazon.com/) which can support upto 32 cores (in which case simply distributing work amongst multiple cores won't work and there needs to be some sort of agglomeration of events. I am also tempted to do the following:

* move away from [NumPy](http://www.numpy.org/) to [Tensorflow](https://www.tensorflow.org/)
* identify potential hotspots and move them to Amazon GPU/FPGA instances.
* play around with different data types and investigate their affect on train and test accuracy.

## Applying Non-Linear Transformation

The non-linear transformation is taken to be each genre multipled with the rest of the genres in the dataset. <br> 
The following figure shows the correlation coefficients between different genres in the dataset.

<p align="center">
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/correlation_coefficients.png">
</p>

It is worth noting that the correlation coefficient between the genre 'Comedy' and 'Drama' is **-0.61976** which shows most of the movies that contain 'Comedy' do not contain 'Drama', and vice versa. <br> The original features remain unchanged and the transformed features are appended to the feature vector, yielding a new feature vector with dimension of **172**.

## Error modelling
Some important assumptions need to be verified without which the model can be misleading. Two of these are [Homoscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity) and [Normality Test](https://en.wikipedia.org/wiki/Normality_test), ideally there should be no be correlations between the residuals and x (independent variable) and the residuals showed be normally distributed.

<div>
    <img = src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/error_modelling.png">
</div>

From the diagram above, we can clearly see that both of the conditions are satisfied. Furthermore I use the [Bartlett Test](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.bartlett.html) to confirm the Homoscedasticity of data, where the samples in predicted values are evenly split into groups of 10. This has resulted in a test statistic equal to **7.016e-12** and a p-value equal to **1**. This we can accept the hypothesis that that the residuals and **x** (the independent variable) are uncorreleated. <br>

## Results

The following figures show the exponentially weighted training and test errors for 671 netflix users which makes it more convieniant to capture the trends in the train and test bias. The value of beta chose for the analysis is 0.9 which is similar to taking the mean over the last 10 iterations. <br>

<div>
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/bias_against_K.png", width="400" height="400" />
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/variance_against_K.png", width="400" height="400" />
</div>

<p align="center">
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/total_error.png", width="400" height="400" />
</p>

<div>
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/test_train_error.png", width="400" height="400" />
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/test_train_regularized.png", width="400" height="400" />
</div>

<br>

<p align="center">
    <img src="https://github.com/aa18514/machine_learning/blob/master/netflix_regression/images/non-linear-features.png" width="400" height="400" />
</p>

| | [RMSE Test](https://en.wikipedia.org/wiki/Root-mean-square_deviation) | [RMSE Train](https://en.wikipedia.org/wiki/Root-mean-square_deviation) | [Mean Test Variance](https://en.wikipedia.org/wiki/Variance) | [Mean Train Variance](https://en.wikipedia.org/wiki/Variance) | [R<sup>2</sup> Score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) | time (s) | [cores](https://en.wikipedia.org/wiki/Multi-core_processor)
| :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Unregularized (Original Features)** | 1.39022 | 0.75140 | 2.416220 | 0.130706 | -0.0287 | 0.636 | 1 |
| **Regularized (Original Features)** | 1.37051 | 0.7478 | 2.3075 | 0.1256 | -0.0195 | 564.307 | 7 |
| **Regularized (Transformed Features)** | 1.28047 | 0.5522 | 1.03005 | 0.068069 | -0.143 | 1910.204 | 7 |

