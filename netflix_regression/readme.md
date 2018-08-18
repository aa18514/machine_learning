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

The loss function in this case is taken to be the [L2 norm](http://mathworld.wolfram.com/L2-Norm.html) <br>

Added support for the multiprocessing module to parallelize against different values of K respectively <br>
