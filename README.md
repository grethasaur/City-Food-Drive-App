# City-of-Edmonton-Food-Drive-App
This includes the Food Drive App with 2 predictive models.


## Overview
This Food Drive App is a project developed as part of the Community Work Integrated Learning program. It serves as a tool to facilitate and enhance food drive initiatives by predicting donation bag collection and volunteer allocation using a proprietary machine learning model. The app provides various functionalities for both organizers and community volunteers.

## Features

### Machine Learning Model
- **Prediction Engine**: The heart of the app, utilizing a custom-built machine learning models to predict the number of donation bags that will be collected as well as adult volunteer based on user inputs.

### User Interface
- **Dashboard**: An interactive dashboard providing a comprehensive view of donation predictions, statistics, and user inputs.
- **Bag prediction**: An interactive predictive model that predicts bag count based on user input.
- **Volunteer prediction**: An interactive predictive model that predicts adult volunteer based on user input.
- **Neighborhood Mapping**: A visual representation of neighborhoods to aid volunteers in planning and execution.
- **Data Collection Form**: Accessible through a link, allowing volunteers to input necessary data for prediction.


## Deployment
This app is currently hosted at [[Link to App Deployment](https://city-food-drive-app-ahdqf2kdsc92abcve5dg9i.streamlit.app/)].
The app in use is, ward_app.py

## Technologies Used
- Python (Pandas, Seaborn,Matplotlib, Scikit-learn)
- Google Maps API and OpenStreetMap (for neighborhood mapping)
