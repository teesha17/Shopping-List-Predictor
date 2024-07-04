import streamlit as st
import pandas as pd
from apyori import apriori

st.set_page_config(page_title="Market Basket Optimisation", layout="centered")
# Load the dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
    transactions = []
    for i in range(0, 7501):
        transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    return transactions

transactions = load_data()

# Train the apriori model
@st.cache_resource
def train_apriori(transactions):
    rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
    results = list(rules)
    return results

rules = train_apriori(transactions)

# Function to inspect results
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(rules), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Streamlit app setup

st.header("Market Basket Optimisation")
st.subheader("Enter an ingredient to see what is often bought with it")

# User input
ingredient = st.text_input("Ingredient")

# Display results
if ingredient:
    st.subheader("Items commonly bought with " + ingredient)
    filtered_results = resultsinDataFrame[resultsinDataFrame['Left Hand Side'] == ingredient]
    if not filtered_results.empty:
        right_hand_side_values = filtered_results['Right Hand Side'].tolist()
        
        # Convert list to comma-separated string
        right_hand_side_string = ", ".join(right_hand_side_values)
        
        # Display the comma-separated string
        st.write(right_hand_side_string)
    else:
        st.write("No commonly bought items found for this ingredient.")

