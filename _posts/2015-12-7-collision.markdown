---
layout: post
title: Modelling injury rates in NYPD Collision data
categories: jekyll update
published: true
markdown: kramdown
---




I use the NYPD collision data from 2013-2014 to illlustrate some machine learning techniques, and see what insights can be gained from modeling the data.

## K-means clustering
The data includes coordinates for each collision, which will be used more easily by learning algorithms if we can group these into a few neighborhoods. I use K-means clustering to put these into a few groups. To determine the optimal number of clusters, I use the 'elbow method' : if you look at the sum of squared distances (of all point to their respective centers) the explained variance as a number of clusters, it decreases rapidly until the optimal point and thereafter the gains are marginal, as shown below.

![Number of clusters vs Inertia]({{site.baseurl}}/_posts/elbow.jpeg)

Clearly this is a bit subjective. The [gap static](https://web.stanford.edu/~hastie/Papers/gap.pdf) is unambiguous and better motivated theoretically, but I use something quick and easy for now. The final cluster centers overlayed over a scatterplot of all the injury-causing collisions:

![clusters.jpeg]({{site.baseurl}}/_posts/clusters.jpeg)

You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve --watch`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

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
