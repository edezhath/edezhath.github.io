---
layout: post
title: Modelling injury rates in NYPD Collision data
categories: jekyll update
published: true
markdown: kramdown
---





I use the NYPD collision data from 2013-2014 to illlustrate some machine learning techniques, and see what insights can be gained from modeling the data.

## K-means clustering
The data includes coordinates for each collision, which will be used more easily by learning algorithms if we can group these into a few neighborhoods . I use K-means clustering, which is a form of the Expectation-Maximization (`E-M`) method. In the `E` step, each point is assigned to the cluster whose "center" is closest. In the `M` step, each cluster center is recomputed as the centroid of all the points that belong to it. These two steps are iterated until you meet a pre-defined tolerance for the distances.

To determine the optimal number of clusters, I use the 'elbow method' : if you look at the sum of squared distances (of all point to their respective centers) the explained variance as a number of clusters, it decreases rapidly until the optimal point and thereafter the gains are marginal, as shown below.

![Number of clusters vs Inertia]({{site.baseurl}}/_posts/elbow.jpeg)

The optimal number of clusters is a bit subjective using this method, since you have to eyeball it. The [gap static](https://web.stanford.edu/~hastie/Papers/gap.pdf) is unambiguous and better motivated theoretically. K-means uses a "hard" partitioning of the points, whereas [Gaussian Mixture models](https://en.wikipedia.org/wiki/Mixture_model) assign a probability or "weight" for belonging to a particular cluster. The final cluster centers overlayed over a scatterplot of all the injury-causing collisions:

![clusters.jpeg]({{site.baseurl}}/_posts/clusters.jpeg)

##Encoding categorical features
We have to encode categorical data into numerical features. We could assign an integer to each value, say `[0,1,2]` for `['BRONX', 'QUEENS', 'BROOKLYN']`. But this would introduce a spurious scale into the data : Brooklyn is not actually twice Queens.  To avoid this, each category is encoded as a seperate 0-1 feature. As shown below, I make a dictionary from the unique values of each feature, and use this to replace them in the Pandas dataframe. You can then run this through scikit-learn's "OneHotEncoder". It's useful to store this encoding dictionary instead of doing it on the fly, since we'll need to use the same encoding scheme for test/future data that we want to make predictions for.
{% highlight python %}
replacedict=dict.fromkeys(data.columns.values)
for i in train.columns.values:
    if(data[i].dtype=='O'):
        uniques=data[i].unique()
        replacedict[i]=dict(zip(uniques,range(len(uniques))))
        data[i]=train[i].replace(replacedict[i])
{% endhighlight %}

##Injury classification/prediction
I tried a few different classification algorithms to predict whether a given collision results in injury. We use AUC in addition to the accuracy to evaulate models, since for an imbalanced dataset like this one, a model can achieve very high accuracy just by predicting every point to be in the same class. Gradient boosted trees (using [xgboost](https://xgboost.readthedocs.org/en/latest/) performed the best)

| Algorithm              | Accuracy | AUC  | Training time |
|------------------------|----------|------|---------------|
| Gradient boosted trees | 0.854    | 0.69 | ~10 mins      |
| Random Forest          | 0.824    | 0.61 | ~5 mins       |
| Logistic Regression    | 0.842    | 0.66 | ~5 seconds    |
| Neural Network         | 0.848    | 0.67 | ~10 mins      |

The GBM does only marginally better than the Logistic Regression, but at a huge premium in computational complexity. ML algorithms generally tend to overfit, so    we have to impose some regularization penalty into the loss function or early stopping criteria to avoid this. Here I use an ElasticNet penalty, meaning a combination of least squared and absolute distances for logistic regression. For the NN I tried both Dropout (randomly omitting some hidden units) 