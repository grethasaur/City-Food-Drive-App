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

    # Stake options
    stake_options = [
        'Bonnie Doon Stake',
        'Edmonton North Stake',
        'Gateway Stake',
        'North Edmonton',
        'Riverbend Stake',
        'YSA Stake'
        ]

    # MultiRoute options
    multiroute_options = ['MultiRoute_No', 'MultiRoute_Yes']

    # Create dropdowns for user selection
    selected_stake = st.selectbox('Select Stake', stake_options)
    selected_multiroute = st.selectbox('Select MultiRoute', multiroute_options)

    # Numerical features
    routes_completed = st.slider("Routes Completed", 1, 10, 5) #CompletedRoutes
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100) #RouteDoors
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)


    # Map selected options to one-hot encoded values
    # Stake mapping
    stake_mapping = {
        'Bonnie Doon Stake': [1, 0, 0, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
        'Edmonton North Stake': [0, 1, 0, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
        'Gateway Stake': [0, 0, 1, 0, 0, 0],  # Replace with the corresponding one-hot encoded values
        'North Edmonton': [0, 0, 0, 1, 0, 0],  # Replace with the corresponding one-hot encoded values
        'Riverbend Stake': [0, 0, 0, 0, 1, 0],  # Replace with the corresponding one-hot encoded values
        'YSA Stake': [0, 0, 0, 0, 0, 1]  # Replace with the corresponding one-hot encoded values
        }

    # MultiRoute mapping
    multiroute_mapping = {
        'MultiRoute_No': [1, 0],  # Replace with the corresponding one-hot encoded values
        'MultiRoute_Yes': [0, 1]  # Replace with the corresponding one-hot encoded values
        }


    # Get one-hot encoded values for selected options
    one_hot_encoded_stake = stake_mapping[selected_stake]
    one_hot_encoded_multiroute = multiroute_mapping[selected_multiroute]



    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('linear_regression_bag.pkl')

        # Prepare input data for prediction, make sure to only put trained features
        # X = [['Stake','MultiRoute','CompletedRoutes','RouteDoors','OverallTime']] from modelling
        numerical_inputs = [routes_completed,doors_in_route, time_spent]
        
        input_data = numerical_inputs + one_hot_encoded_stake + one_hot_encoded_multiroute

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

# Page 6: Chatbot
import os
# Set the environment variables from the st.secrets dict
os.environ['GRADIENT_ACCESS_TOKEN'] = st.secrets["GRADIENT_ACCESS_TOKEN"]
os.environ['GRADIENT_WORKSPACE_ID'] = st.secrets["GRADIENT_WORKSPACE_ID"]

from gradientai import Gradient

def chatbot():
    # Streamlit title and description
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")


        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            samples = [
            { "inputs": "### Instruction: Provide information about the food drive event. \n\n### Response: The food drive event is a community initiative aimed at collecting food items for those in need. We encourage participants to donate non-perishable food items such as canned goods, rice, pasta, and other essential items. Your contributions will make a difference in the lives of individuals and families facing food insecurity." },
            { "inputs": "### Instruction: What is the purpose of the food drive campaign? \n\n### Response: The food drive campaign aims to gather food donations to support local communities in need. By donating non-perishable items, you are helping to alleviate hunger and improve the lives of those facing food shortages. Your generous contributions are vital in making this initiative successful." },
            { "inputs": "### Instruction: Tell me more about how the food drive works. \n\n### Response: The food drive operates by encouraging individuals and organizations to donate non-perishable food items. These items can be dropped off at designated collection points or picked up by volunteers on specified dates. All collected food items are then distributed to local food banks, shelters, and needy families. Your participation in the food drive will contribute to ensuring that everyone in our community has access to nutritious meals." },
            { "inputs": "### Instruction: How can I get involved in the food drive? \n\n### Response: Getting involved is easy! You can volunteer your time by helping with collections, spreading awareness, or even organizing your own donation event. Every effort counts toward supporting our community members in need." },
            { "inputs": "### Instruction: What impact will my donation have on the community? \n\n### Response: Your donation directly supports local families and individuals facing food insecurity. By contributing non-perishable items, you're providing essential sustenance and hope to those who need it most." },
            { "inputs": "### Instruction: Where can I drop off my donations for the food drive? \n\n### Response: You can drop off your donations at designated collection points across the city. Check our website or contact us for specific locations and hours. Your contributions will go a long way in helping our cause!" },
            { "inputs": "### Instruction: Can my company/organization participate in the food drive? \n\n### Response: Absolutely! We welcome corporate or organizational involvement. You can organize a donation drive within your workplace or collaborate with us to make a substantial impact on our community's well-being." },
            { "inputs": "### Instruction: What specific food items are most needed for donations? \n\n### Response: While all donations are appreciated, items like canned vegetables, protein-rich foods (canned meat, peanut butter), grains, and hygiene products are highly sought after. Your donations of these items will help meet varied nutritional needs." }
            ]
      
            #fine tuning
            num_epochs = 3
            count = 0
            while count < num_epochs:
              new_model_adapter.fine_tune(samples=samples)
              count = count + 1

            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # after fine-tuning
            complete_response = new_model_adapter.complete(
                query=sample_query,
                max_generated_token_count=100)
            st.markdown(f"Generated: {complete_response.generated_output}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()


# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "Neighbourhood Mapping", "Data Collection", "Chatbot"])

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
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
