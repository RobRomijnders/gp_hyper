# What does the log likelihood surface of GP like?

The title explains the entire blogpost. I am preparing for the Gaussian Processes Summer School in September 2018 in Sheffield.

Now, I wondered how the log-likelihood changes when I change the hyperparameters. So I pulled some Kaggle code from the internet and copied some Sklearn code from their website. Just to be clear: I wrote few lines of code myself here. Then I use pyplot to make a contour plot when changing two of the important hyperparameters.

Some details:
  
  *  **House pricing data set** I want a simple data set that everyone uses for basic regression tasks. It seems that every intro-to-ml course uses the house pricing data set at some stage. [Link here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  * **preprocessing** I pulled the first kernel that I could find on Kaggle. I forgot the link.
  * **Implementation** I use the default [Gaussian Processes Regressor](http://scikit-learn.org/stable/modules/gaussian_process.html) from Sklearn

# What hyperparameters to consider?

After reading Rasmussen's book on Gaussian Processes ([pdf link](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)), I concluded that the squared exponential kernel seems a default choice. The kernel looks like this:

<img alt="$k(x_i, x_j) = c \ e^{\frac{(x_i - x_j)^2}{l}}$" src="https://rawgit.com/RobRomijnders/gp_hyper/master/svgs/ffbc1c03322287368db5af7a69cf2e6e.svg?invert_in_darkmode" align=middle width="152.80996499999998pt" height="45.03839999999998pt"/>

This kernel has two hyperparameters:
  
  * <img alt="$c$" src="https://rawgit.com/RobRomijnders/gp_hyper/master/svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width="7.087278000000003pt" height="14.102549999999994pt"/>, which is also named the covariance scale
  * <img alt="$l$" src="https://rawgit.com/RobRomijnders/gp_hyper/master/svgs/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode" align=middle width="5.208868500000004pt" height="22.745910000000016pt"/>, which is also named the length scale

# Show me the pictures

Changing the hyperparameters and plotting the log likelihood on the validation data gives the following diagram:

![image](https://www.google.nl/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjJz7KAir3cAhUDjqQKHRZaDIAQjRx6BAgBEAU&url=http%3A%2F%2Fdragonballfanon.wikia.com%2Fwiki%2FFile%3ARandom.png&psig=AOvVaw2PYrEE6379K6w8BOS6tNDh&ust=1532705213052023)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com

