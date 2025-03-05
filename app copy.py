import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        background-color: #87CEEB;  /* Sky blue background */
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E90FF;  /* Deep sky blue */
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #4682B4;  /* Steel blue */
        font-size: 18px;
        text-align: center;
        margin-bottom: 40px;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.9);  /* White with transparency */
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the Page
st.markdown('<div class="main-header">✈️ Flight Ticket Price Prediction</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="sub-header">Analyze and predict flight ticket prices with ease!</div>', unsafe_allow_html=True)

# Add an Image
st.image(
    "https://t3.ftcdn.net/jpg/00/20/13/60/360_F_20136083_gk0ppzak6UdK9PcDRgPdLjcuAdo7o1LK.jpg",  
    caption="Predict the prices of flight tickets efficiently!",
    use_container_width=True
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline_tuned.pkl")

model = load_model()

# Sidebar Inputs
st.sidebar.header("Input Flight Details")
airline = st.sidebar.selectbox("Select Airline", ["Indigo", "Air India", "SpiceJet", "Vistara", "GoAir", "Jet Airways"], key="airline")
source_city = st.sidebar.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata"], key="source_city")
destination_city = st.sidebar.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata"], key="destination_city")
travel_class = st.sidebar.radio("Travel Class", ["Economy", "Business"], key="travel_class")
departure_time = st.sidebar.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night"], key="departure_time")
arrival_time = st.sidebar.selectbox("Arrival Time", ["Morning", "Afternoon", "Evening", "Night"], key="arrival_time")

# Valid Stops sesuai dengan dataset
valid_stops = ["zero", "one", "two_or_more"]
stops = st.sidebar.radio("Number of Stops", valid_stops, key="stops")

duration = st.sidebar.slider("Duration (hours)", min_value=1, max_value=24, value=2, key="duration")
days_left = st.sidebar.slider("Days Left to Departure", min_value=0, max_value=365, value=30, key="days_left")

# Create DataFrame for Prediction
input_data = pd.DataFrame({
    "airline": [airline],
    "source_city": [source_city],
    "destination_city": [destination_city],
    "class": [travel_class],
    "departure_time": [departure_time],
    "arrival_time": [arrival_time],
    "stops": [stops],  # Updated to match valid_stops
    "duration": [duration],
    "days_left": [days_left]
})

# Predict Flight Price
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Ticket Price")
    st.markdown(
        f"""
        <div style="font-size: 24px; font-weight: bold; color: #008000;">
            💵 Estimated Price: ₹{prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Input Data
st.header("Your Input Flight Details")
st.write(input_data)

# Data Visualization
st.header("Ticket Price Analysis")
@st.cache_data
def load_data():
    # Replace with your actual dataset
    return pd.read_csv("flight_data.csv")

df = load_data()

# Data Visaulization

# Price Distribution by Class
st.subheader("Flight Ticket Price Distribution by Class")

# Separate the data for business and economy classes
df_business = df[df['class'] == 'Business']
df_economy = df[df['class'] == 'Economy']

# Visualize the distribution for Business class
st.markdown("**Business Class**")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_business['price'], kde=True, bins=30, color='blue', ax=ax)
ax.set_title('Distribution of Flight Ticket Prices (Business Class)')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Visualize the distribution for Economy class
st.markdown("**Economy Class**")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_economy['price'], kde=True, bins=30, color='green', ax=ax)
ax.set_title('Distribution of Flight Ticket Prices (Economy Class)')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Insight
st.markdown(
    """
    **Insights**:
    - The ticket prices for Business Class are widely spread and generally higher compared to Economy Class.
    - Economy Class has a more concentrated distribution with lower ticket prices.
    - This reflects the expected pricing structure, where Business Class tickets are generally more expensive.
    """
)

# Airline prices based on the class and company
st.subheader("Airline Prices Based on the Class and Company")

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=df, 
    x="airline", 
    y="price", 
    hue="class", 
    palette="viridis", 
    ax=ax
)
ax.set_title("Airline Prices Based on the Class and Company")
ax.set_xlabel("Airline")
ax.set_ylabel("Price")
st.pyplot(fig)

# Insight
st.markdown(
    """
    **Insights**:
    - Business class tickets are significantly more expensive than economy class tickets for airlines like Air India and Vistara.
    - Economy class ticket prices are relatively consistent across airlines.
    - Airlines like Indigo and AirAsia primarily have lower ticket prices, indicating they target cost-conscious travelers.
    """
)



# Scatter Plot: Duration vs Price by Class
st.write("### Duration vs Price by Class")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="duration",
    y="price",
    hue="class",
    palette={"Economy": "blue", "Business": "orange"},
    alpha=0.6,
    ax=ax2
)
ax2.set_title("Duration vs Price by Class")
ax2.set_xlabel("Duration (hours)")
ax2.set_ylabel("Price")
st.pyplot(fig2)
st.write(
    """
    **Insight**: Business class tickets generally have higher prices regardless of duration.
    For economy class, ticket prices remain relatively low and show minimal variation, 
    even for longer flights.
    """
)

# Additional Visualization: Average Price by Class
st.write("### Average Price by Class")
avg_price_class = df.groupby("class")["price"].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=avg_price_class, x="class", y="price", palette={"Economy": "blue", "Business": "orange"}, ax=ax3)
ax3.set_title("Average Ticket Price by Class")
ax3.set_xlabel("Class")
ax3.set_ylabel("Average Price")
st.pyplot(fig3)
st.write(
    """
    **Insight**: The average ticket price for business class is significantly higher than 
    economy class, confirming the premium nature of business class travel.
    """
)

# Heatmap Correlation
st.subheader("Numerical Feature Correlation")
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    df[["duration", "days_left", "price"]].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    ax=ax
)
ax.set_title("Correlation Numerical Features")
st.pyplot(fig)

# Insight
st.markdown(
    """
    **Insights**:
    - The correlation between `duration` and `price` is **0.20**, indicating a weak positive relationship. Longer flight durations tend to have slightly higher prices.
    - `days_left` and `price` have a weak negative correlation of **-0.09**, reflecting airline pricing strategies where booking earlier often results in lower prices
    """
)
# Add Actual vs Predicted Prices Visualizations to Streamlit
st.subheader("Comparison of Actual vs Predicted Prices")

# Display the Training Set Image
st.markdown("**Training Set**")
st.image("train_comparison.png", caption="Comparison of Actual vs Predicted Prices (Training Set)", use_container_width=True)

# Display the Test Set Image
st.markdown("**Test Set**")
st.image("test_comparison.png", caption="Comparison of Actual vs Predicted Prices (Test Set)", use_container_width=True)

# Add Insights Markdown
st.markdown(
    """
    ### Insights from Actual vs Predicted Prices

    #### Training Set
    The scatterplot comparing actual and predicted prices for the training set shows that most points align closely with the red dashed line (ideal line). This indicates that the model performs well in capturing patterns from the training data. The tight clustering around the line suggests high accuracy in predictions.

    #### Test Set
    For the test set, the scatterplot also demonstrates good alignment between actual and predicted prices. While there is slightly more spread compared to the training set, the predictions remain accurate and follow the general trend of the actual prices. This reflects the model's ability to generalize well to unseen data.

    ### Strengths and Weaknesses
    - **Strengths**:
    - High alignment of actual and predicted prices indicates strong model performance.
    - Minimal overfitting is evident as the training and test performance are consistent.

    - **Weaknesses**:
    - Slight underestimation and overestimation in some areas, particularly for extreme values, suggest room for improvement in handling outliers or rare cases.

    ### Business Connection
    Accurate predictions like these allow businesses (airlines and travel platforms) to better anticipate market pricing trends, ensuring competitive pricing strategies while maintaining profitability. Travelers can also benefit by identifying the best times and conditions to book tickets for cost efficiency.
    """
)

# Footer
st.markdown("---")
st.markdown("**Note**: This is a sample prediction system and may vary from real-world prices.")
