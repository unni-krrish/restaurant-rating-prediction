# Prediction of restaurant ratings with simple data pipeline.
## Description
In this experiment, I attempt to predict ratings of restaurants listed on [Zomato](https://www.zomato.com/) in the city of Bangalore. This challenge was adapted from [kaggle](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants). The challenge is to predict rating using columns such as restaurant type, cuisines offered, location, etc. Apart from achieving a prediction model of good enough accuracy, a data pipeline has also been built to predict the rating of new restaurants. 
## How to run the pipeline?
Two samples are provided - [execution.py](https://github.com/unni-krrish/restaurant-rating-prediction/blob/main/execution.py) and [sample_prediction.py](https://github.com/unni-krrish/restaurant-rating-prediction/blob/main/sample_prediction.py). Both training of the model as well as prediction on unseen test points are demonstrated. 
## Added functionalities:
- Autosave last trained model and use it for predicting unseen test points.
- Take raw csv file as input and perform cleaning, feature encoding and modelling without supervision.

## Main takeaways from EDA
- The relationship between number of votes and rating is not entirely independent. Restaurants when rated high consistently, tend to receive many votes from users. However, it still remains as a fact that if a restaurant has large number of votes, the rating is likely to be high. 
- When offering exotic cuisines, retaurants are likely to be rated higher. Indian restaurants offering Indian cuisines tend to receive an average rating slightly below 4. This could be because of the variety of meals and customer's expectations regarding how a dish should be prepared. 
- For new restaurants, the restaurant type would already be fixed. So to increase the rating, it can consider offering more variety in the cuisines and offering services like table booking and optional online ordering. Serving the meals to a focused customer group is highly recommended as the rating is severely affected by how large the audience is.

## Data/Model Pipeline 
![Picture1](https://user-images.githubusercontent.com/53073761/135336015-33d68146-8e02-4520-ac1d-0ad2130741db.png)



## Feature Importances at a glance (SHAP Summary)
![img](https://user-images.githubusercontent.com/53073761/134924384-97b9a66b-4820-44e8-a82d-56a7079144ce.jpg)
