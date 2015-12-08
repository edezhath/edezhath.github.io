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
We have to encode categorical data into numerical features. We could assign a numerical value to each feature, say `[0,1,2]` for `['BRONX', 'QUEENS', 'BROOKLYN']`. But this would introduce an unintended scale into the data : Brooklyn is not actually twice Queens.  To avoid this, each category is encoded as a seperate 0-1 feature. 

{% highlight python %}
dist=[]
for i in range(5,60,3):
    kmn=KMeans(n_clusters=i, init='k-means++', n_init=5, max_iter=300, tol=0.000001, precompute_distances=True, n_jobs=3)
    kmn=kmn.fit(locdata)
    cent=kmn.cluster_centers_
    dist.append([i,kmn.inertia_])
#=> Intertia graph
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
