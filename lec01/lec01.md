#### Python Resources

* [Google Python Class](https://developers.google.com/edu/python/): If you have never coded in python before or need a refresher. 

* Stanford also released a great [resource for learning python/numpy](http://cs231n.github.io/python-numpy-tutorial/)


#### Linear Algebra Resources
* [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

* [Khan Academy](https://www.khanacademy.org/math/linear-algebra): A great series of short videos on key concepts on linear algebra

* [MIT OCW Linear Algebra by Prof. Gilbert Strang](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/): This is by far the best resource available for Linear Algebra. Be careful though you might spend lot of time on these lectures as they are amazing!


#### Tensorflow
Please install [Tensorflow v1.9](https://www.tensorflow.org/install/). We recommend installation using Docker or Conda environment

* I will post here exact steps I tried to install tensorflow on Macbook using conda
  * First, [install anaconda](https://www.anaconda.com/download/#macos)
  * Create a conda environment, activate it and install tensorflow

    ```
    conda create -n tf python=3.6
    source activate tf
    (tf) pip install -r ../requirements.txt
    ```
  * Note, as of this post python=3.7 *does not work* with tensorflow
