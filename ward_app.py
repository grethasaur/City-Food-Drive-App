import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load the dataset with a specified encoding
data = pd.read_csv('filled_df.csv', encoding='latin1')
data_cleaned = data.drop(columns = ["Unnamed: 0", "Comments"])

# Page 1: Dashboard
def dashboard():
    st.image('Logo.png', use_column_width=True)

    st.subheader("💡 Abstract:")

    inspiration = '''
    The Edmonton Food Drive Project....Talk about the Project here and lessons learned
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    Your project description goes here.
    '''

    st.write(what_it_does)


# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")


    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data_cleaned, x='AdultVolunteer', nbins=20, labels={'AdultVolunteer': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='YouthVolunteer', nbins=20, labels={'YouthVolunteer': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='Bags', nbins=20, labels={'Bags': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data_cleaned, x='RouteTime', nbins=20, labels={'RouteTime': 'Time to Complete(min)'})
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data

    # # Stake options
    # stake_options = [
    #     'Bonnie Doon Stake',
    #     'Edmonton North Stake',
    #     'Gateway Stake',
    #     'North Edmonton',
    #     'Riverbend Stake',
    #     'YSA Stake'
    #     ]

    # MultiRoute options
    multiroute_options = ['MultiRoute_No', 'MultiRoute_Yes']

    # List of available Ward options
    ward_options = [
        'Beaumont Ward',
        'Belmead',
        'Blackmud Creek Ward',
        'Clareview Ward',
        'Connors Hill Ward',
        'Coronation Park Ward',
        'Crawford Plains Ward',
        'Drayton Valley Ward',
        'Ellerslie Ward',
        'Forest Heights Ward',
        'Greenfield Ward',
        'Griesbach Ward',
        'Lee Ridge Ward',
        'Londonderry Ward',
        'Namao Ward',
        'Rabbit Hill Ward',
        'Rio Vista Ward',
        'Rutherford Ward',
        'Silver Berry Ward',
        'Southgate Ward',
        'Stony Plain Ward',
        'Strathcona Married Student Ward',
        'Terwillegar Park Ward',
        'Wainwright Branch',
        'Wild Rose Ward',
        'Windsor Park YSA Ward',
        'Woodbend Ward'
        ]


    # Create dropdowns for user selection
    # selected_stake = st.selectbox('Select Stake', stake_options)
    selected_ward = st.selectbox('Select Ward', ward_options)
    selected_multiroute = st.selectbox('Select MultiRoute', multiroute_options)

    # Numerical features
    routes_completed = st.slider("Routes Completed", 1, 10, 5) #CompletedRoutes
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100) #RouteDoors
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)


    # Map selected options to one-hot encoded values
    # Stake mapping
    # stake_mapping = {
    #     'Bonnie Doon Stake': [1, 0, 0, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
    #     'Edmonton North Stake': [0, 1, 0, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
    #     'Gateway Stake': [0, 0, 1, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
    #     'North Edmonton': [0, 0, 0, 1, 0, 0],  # Replace with the corresponding one-hot encoded values
    #     'Riverbend Stake': [0, 0, 0, 0, 1, 0],  # Replace with the corresponding one-hot encoded values
    #     'YSA Stake': [0, 0, 0, 0, 0, 1]  # Replace with the corresponding one-hot encoded values
    #     }

    # MultiRoute mapping
    multiroute_mapping = {
        'MultiRoute_No': [1, 0],  # Replace with the corresponding one-hot encoded values
        'MultiRoute_Yes': [0, 1]  # Replace with the corresponding one-hot encoded values
        }

    #Ward mapping
    ward_mapping = {selected_ward: [1 if selected_ward == ward else 0 for ward in ward_options] for selected_ward in ward_options}


    # Get one-hot encoded values for selected options
    # one_hot_encoded_stake = stake_mapping[selected_stake]
    one_hot_encoded_multiroute = multiroute_mapping[selected_multiroute]
    one_hot_encoded_ward = ward_mapping[selected_ward]


    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('linear_regression_ward_only.pkl')

        # Prepare input data for prediction, make sure to only put trained features
        # X = [['Stake','MultiRoute','CompletedRoutes','RouteDoors','OverallTime']] from modelling
        numerical_inputs = [routes_completed,doors_in_route, time_spent]
        
        input_data = numerical_inputs + one_hot_encoded_ward + one_hot_encoded_multiroute

        # Make prediction
        prediction = model.predict([input_data])

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed


# Page 4: Neighbourhood Mapping
# Read geospatial data
geodata = pd.read_csv("EDA2_merged_data.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]
        no_data = (geodata['Neighbourhood'] == neighbourhood) & geodata['Latitude'].isna()
        naur_data = no_data.any()

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty or naur_data == True:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")



# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/Sif2hH3zV5fG2Q7P8"#YOUR_GOOGLE_FORM_URL_HERE
    st.markdown(f"[Fill out the form]({google_form_url})")

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Neighbourhood Mapping", "Data Collection"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()

if __name__ == "__main__":
    main()
