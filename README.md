# California-Housing-Prices-Data-Modeling

![California Housing Prices Data Modeling](https://user-images.githubusercontent.com/71048405/196056650-c8cb09e5-b2ea-494e-8d89-fe0e0b835354.jpeg)


## About Dataset
This is the dataset used in the second chapter of Aurélien Géron's recent book 'Hands-On Machine learning with Scikit-Learn and TensorFlow'. It serves as an excellent introduction to implementing machine learning algorithms because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome.

The data contains information from the 1990 California census. So although it may not help you with predicting current housing prices like the Zillow Zestimate dataset, it does provide an accessible introductory dataset for teaching people about the basics of machine learning.

## Content
The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. Be warned the data aren't cleaned so there are some preprocessing steps required! The columns are as follows, their names are pretty self explanitory:

longitude
latitude
housingmedianage
total_rooms
total_bedrooms
population
households
median_income
medianhousevalue
ocean_proximity
Acknowledgements

## This data was initially featured in the following paper:
Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997): 291-297.

and I encountered it in 'Hands-On Machine learning with Scikit-Learn and TensorFlow' by Aurélien Géron.
Aurélien Géron wrote:
This dataset is a modified version of the California Housing dataset available from:
Luís Torgo's page (University of Porto)

## Inspiration
See my kernel on machine learning basics in R using this dataset, or venture over to the following link for a python based introductory tutorial: https://github.com/ageron/handson-ml/tree/master/datasets/housing


# California Housing Prices Data Modeling

* [importing the libraries](#importing-the-libraries)
* [Reading the data](#Reading-the-data)
* [Exploring the data](#Exploring-the-data)
* [Visualizing the data](#Visualizing-the-data)
* [Drop duplicates values](#Drop-duplicates-values)
* [Missing data](#Missing-data)
* [Filling in missing values](#Filling-in-missing-values)
* [Outlier and Deleting Observations](#Outlier-and-Deleting-Observations)
* [Encoding categorical features](#Encoding-categorical-features)
* [Drop unimportant columns](#Drop-unimportant-columns)
* [Scaling and Split the data](#Scaling-and-Split-the-data)
* [Linear Regression](#Linear-Regression)
* [Ridge regression](#Ridg-regression)
* [Lasso Regression](#Lasso-Regression)
* [compersion between models](#compersion-between-models)


