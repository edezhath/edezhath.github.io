---
published: true
layout: post
categories: jekyll update
markdown: kramdown
---



In most machine learning problems, some number of the features are noise or unrelated to the variable we're trying to predict. Reducing these features can greatly improve the accuracy and efficiency of the learning algorithm, and in some cases it's a necessity : when computational resources are too limited to run all the available features through your algorithm. We'll look at how very shallow Random Forests can be used to quickly and easily reduce the number of features without losing useful information. We'll also compare this with PCA, which is a commonly used technique for dimensionality reduction. We'll use the RandomForest and PCA implementations from scikit-learn. The target we'll try to model will be a function of 5 "true" features : 5 columns of numbers randomly drawn from [0, 1).

{% highlight python %}
import sklearn.ensemble, sklearn.decomposition, numpy as np
truefeatures=np.random.rand(1000,5)
{% endhighlight %}

The regression target is generated as the following function of these random variables c0...c4:
$$
f(c_0,c_1,c_2,c_3,c_4)=(c_0+c_1)sin(c_2)-c_3 e^{ c_4}
$$

{% highlight python %}
target=map(lambda c:(c[0]+c[1])*np.sin(c[2])-c[3]*np.exp(c[4]),truefeatures[:,:5])
{% endhighlight  %}

Next we'll add a hundred columns of junk, unrelated to the target

{% highlight python %}
train=np.append(truefeatures,np.random.rand(1000,100),axis=1)
{% endhighlight  %}

We'll make a Random Forest that has 500 trees with 2 levels each and we'll only look at 3 features at a time. Quick intro to Random Forests: it's a collection of decision trees, each trained on a randomly drawn subset of the data and features.

{% highlight python %}
forest=sklearn.ensemble.RandomForestRegressor(n_estimators=500,max_depth=2,max_features=3)
%time forest=forest.fit(train,target)
{% endhighlight  %}

{% highlight ruby %}
CPU times: user 459 ms, sys: 0 ns, total: 459 ms
Wall time: 460 ms
{% endhighlight  %}

Very fast even on my dated i5-3230M. Now we can use the built in feature relevance score to find the ones to keep. The 5 highest ones are as expected, the true features. And they're several standard deviations above the mean score - one's even up to particle physics standards!

{% highlight python %}
np.argsort(forest.feature_importances_)[-5:]
{% endhighlight %}

{% highlight numpy %}
array([4, 3, 0, 1, 2])
{% endhighlight %}


{% highlight python %}
(forest.feature_importances_[:5]-np.mean(forest.feature_importances_))/np.std(forest.feature_importances_)
{% endhighlight %}
``array([ 3.52628814,  2.93327098,  3.89061919,  5.21814166,  3.33544238])``

Is this reliable? Let's do a hundred runs with just 300 trees and count what fraction of times the true features are in the top 5.

{% highlight numpy %}
histrf=np.zeros(5)
forest=sklearn.ensemble.RandomForestRegressor(n_estimators=300,max_depth=2,max_features=3)
for i in range(100):
    truefeatures=np.random.rand(1000,5)
    target=map(lambda c:(c[0]+c[1])*np.sin(c[2])-c[3]*np.exp(c[4]),truefeatures[:,:5])
    train=np.append(truefeatures,np.random.rand(1000,100),axis=1)
    forest=forest.fit(train,target)
    top5=frozenset(np.argsort(forest.feature_importances_)[-5:]).intersection(range(5))
    for j in top5:
        histrf[j]+=1
histrf/100
{% endhighlight %}
    
``array([ 0.83,  0.87,  0.96,  0.97,  0.91])``

The less trees you have, the less likely they are to be sampled enough to be found as good features. 2-3X the number of total features will get you the right ones most of the time, and 4-5X will pretty much guarantee they're the top ones. How does PCA do in this problem? Since all our features are completely independent, we know the principal components will consist of the junk and true features with equal probability. But just for fun, let's look at how often the true features are among the largest five factors in the first five principal components.

{% highlight python %}
pca=sklearn.decomposition.PCA(n_components=5)
histpca=np.zeros(5)
for i in range(100):
    original=np.random.rand(1000,5)
    target=map(lambda c:(c[0]+c[1])*np.sin(c[2])-c[3]*np.exp(c[4]),original[:,:5])
    train=np.append(original,np.random.rand(1000,100),axis=1)
    pca=pca.fit(train)
    top5=frozenset(np.ravel(np.argsort(np.abs(pca.components_))[:,-5])).intersection(range(5))
    for j in top5:
        histpca[j]+=1
histpca/100
{% endhighlight %}

``array([ 0.11,  0.04,  0.04,  0.05,  0.06])``

Quite dismal. But this isn't really fair, since we would know not to use PCA in the case : the variance explained by the top principal components is only a tiny fraction of the total variance in the data.

{% highlight python %}
pca.explained_variance_ratio_[:5]
{% endhighlight %}
``array([ 0.01665517,  0.01635231,  0.01580367,  0.01550839,  0.01541421])``

How much does removing the junk features help with the final training algorithm? To make it easier to score the performance (so we can use area under ROC) we'll convert this into a classification problem. The mean value of the target function is f(ci=0.5)≈−0.344935, so we'll use this as our decision boundary and get an even distribution of 1's and 0's in the target column.

{% highlight python %}
target=map(lambda c:int(((c[0]+c[1])*np.sin(c[2])-c[3]*np.exp(c[4]))>-0.344935),truefeatures[:,:5])
{% endhighlight %}

We'll use a fully connected feed-forward neural network with 2 hidden layers to model this. The code for the network is written in Theano, and is adapted from the MLP example from deeplearning.net. We'll use twice as many hidden units as inputs, set aside 10% of the data as a validation set which will be used for early stopping, and use an L2 norm for regularization.

{% highlight python %}
import trainmlp
model=trainmlp.test_mlp(train,target,0.9,2,n_epochs=1000,learning_rate=.1,n_hidden=[210,210],L2_reg=0.0005,L1_reg=0)
... building the model
... training
{% endhighlight %}

Optimization complete. Best validation score of 3.750000 % obtained at iteration 3420
AUC is 0.992181
The code for file trainmlp.py ran for 0.58m

{% highlight python %}
model=trainmlp.test_mlp(train[:,:5],target,0.9,2,n_epochs=1000,learning_rate=.1,n_hidden=[10,10],L2_reg=0.0005,L1_reg=0)
{% endhighlight %}

... building the model
... training
Optimization complete. Best validation score of 2.500000 % obtained at iteration 
AUC is 0.998354
The code for file trainmlp.py ran for 0.02m

The reduced feature set trains 25X faster, and has a 50% improvement in the error rate! But in this case we knew how many features to keep. So how do we figure out how many to keep in a more realistic scenario? We could shuffle the labels to get a baseline score, and use this to determine which features to keep or discard. I'll look at this in more detail in the next post.
