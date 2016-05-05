## **Summary**
SimilarCarFinder is a Computer Vision based application aimed to help users purchase their next car based on how the car looks. It allows users to better explore a variety of the cars that might be interesting to him. Behind the scenes, the Web App uses a Neural Network engine to find cars on Craigslist that looks similar to an image provided by the user.

## **Data**
I've built an Application to help users to improve the quality of their search on websites that sell cars. For this purpose I scraped the data from Craigslist. Scraper gets the information from the basic search page, stores URLs for each car page and then scrapes information particularly for every car. 

## **Process Flow**

After scraping the images of the cars, I preprocess all of them (reshape) and pass to the Neural Network.
I use convolutional neural network (VGG-CNN) pretrained on ImageNet dataset. I drop the last layer of the original net to use the output as features.
For implementation I use package Lasagne and nolearn class. All the computation are being runned on AWS EC2. 
I've written code for featurizing one image and all the database. To do this more effectively, I added parralel computations block for image preprocessing.

I save all the information about featurized images in pandas dataframe, where I store also links to the webpages of all the cars. 

After that I created two variants of the algorthm to compare user's car to the cars in my database: first uses only one (main) photo of every car, second uses all the images available for every car. In the core of these algorithms there is cosine similarity, which should be the highest for the cars that are the most alike. 

I created web-interface using StartBootstrap and Flask. The user can submit the URL to an image of the car he likes and can see the options for the cars he can buy from Craigslist.

*readme will be expanded*
