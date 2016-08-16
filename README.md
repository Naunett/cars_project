## **Summary**
SimilarCarFinder is a Computer Vision based application aimed to help users purchase their next car based on how the car looks. It allows users to better explore a variety of the cars that might be interesting to him. Behind the scenes, the Web App uses a Neural Network engine to find cars on Craigslist that looks similar to an image provided by the user.

## **Motivation**
About a year ago I was looking for a new car. Craigslist has lots of awesome deals, but it is extremely hard to do search on it! Often the description doesn't correspond to the search query at all. So, if in real life finding a car that looks like you want is not a problem – you see all of them right away at any car dealership, this is not the case for online search. And I decided to create an app that would help me to filter cars based on the photo of the car. 

## **Data**
An app can be adjusted to work with any website that sell cars, but I worked only with Craigslist's data. For this purpose I scraped the data from Craigslist, using Python libraries **BeautifulSoup**, **requests** and **urllib**. </br>
My scraper gets the information from the basic search page, stores URLs for each car page and then scrapes information particularly for every car (I wanted to use all the images for every car listing and I was able to get this information only from individual car pages). I needed to use some tricks with regular epressions to get the right links to the cars images. All the scraped information is stored in **Pandas** dataframe, including links to photos. 

## **Process Flow**

<img src="https://github.com/Naunett/cars_project/blob/master/img/1_pipeline.png" width =70%; height=70% /> </br>

### Preprocessing

After scraping the images of the cars, I reshaped them to 224x224 pixels. The input layer of the convolutional neural network that I used requires images of (3, 224, 224) dimensions. Number 3 corresponds to three RGB channels.

### Featurization and comparisons

To featurize images I use Oxford Visual Geometry Group convolutional neural network (**VGG-CNN**) pretrained on ImageNet dataset. I drop the last layer of the original net and I use the output of the 4096 dimensional deeper dense layer as features.

<img src="https://github.com/Naunett/cars_project/blob/master/img/2_VGG-CNN.png" width =70%; height=70% /> </br>

Original information can be found on their official web-site:
[http://www.robots.ox.ac.uk/~vgg/](http://www.robots.ox.ac.uk/~vgg/)

For implementation I use package **Lasagne** and **Nolearn** class. All the computation I run on **AWS EC2** in GPU mode with Nvidia CUDA. I've written separate code for featurizing one image and all the database. To do this more effectively, I added parralel computations block for image preprocessing.

I save all the information about featurized images in Pandas dataframe, where I store also links to the webpages of all the cars. 

After that I created two variants of the algorthm to compare user's car to the cars in my database: first uses only one (main) photo of every car, second uses all the images available for every car. To compare featurized images I use **cosine similarity**, the highest meaning of which corresponds to the cars that are the most alike. 

### Web application

I created a web-interface using **StartBootstrap** and **Flask**. 
<img src="https://github.com/Naunett/cars_project/blob/master/img/3_search_page.png" width =70%; height=70% /> </br>

The user can submit the URL to an image of the car he likes and can see the options for the cars he can buy from Craigslist, with links. 
Even if the car looks very specific, application can find cars that look alike.
<img src="https://github.com/Naunett/cars_project/blob/master/img/5_truck_example.png" width =70%; height=70% /> </br>

### Notes 
Due to the usage of pre-trained neural network, there are some issues in the work of the application. As an example, originally alorithm wasn't able to find corresponding cars if to put, for example, the photo of front view of the car. I eliminated it by incorporating all the photos available for every car in listings. It just doesn't know that those are same cars. Second big issue is the color. Color feature is considered by the neural net to be more important than shape of the car. It tends to find cars of the same colors, rather than same models. To improve the quality of the recommendations, fine-tuning of neural network using specific cars dataset is necessary.

### Code
The folder neural_net_code contains two files: </br>
- **nolearn_load_weights.py** loads weights of neural net. It has a code to create neural network of specific architecture in Nolearn, set weights and save in pickled file.
- **initialize_net.py** initializes neural net. Can be used to initialize neural net using file with weights.

The folder similar_car is the main folder that contains the application.
In src folder all the main code files can be found:
- **craigslist_car_scraper.py** is a custom-made web scraper for Craigslist.
- **img_featurizer.py** contans code to featurize images of the cars.
- **db_featurizer.py** contains code (using object oriented approach) to featurize the data from Craigslist. It needs to be run once and after that information about featurized images is stored in a pickle file.
- **predict.py** contains code to compare user's image to all the cars images in a database.

**app.py** runs Flask web application.


