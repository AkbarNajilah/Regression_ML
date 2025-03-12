# ✈️Flight Ticket Price Prediction Project

## Justification
1. Predicting ticket prices is a significant consumer need, with studies showing that travelers are highly price-sensitive when planning trips:
    - [How Flight Prices Affect Travel Decisions](https://www.travelpulse.com/news/impact-of-flight-pricing-on-consumer-behavior)
2. Airlines and travel platforms can benefit from optimized pricing strategies to improve customer satisfaction and increase revenue:
    - [The Role of Dynamic Pricing in Aviation](https://aviationeconomics.com/articles/dynamic-pricing-and-its-impact-on-airlines)

By addressing these needs, the project combines predictive modeling with business impact, bridging the gap between consumer affordability and airline revenue optimization.

## Problem Background
Flight ticket pricing is a complex and dynamic process influenced by several factors, such as travel dates, duration, stops, and airline operators. Predicting flight ticket prices is crucial for both consumers and businesses. Consumers want to book flights at the best price, while airlines and travel platforms strive to optimize pricing strategies to maximize revenue.

## Project Output
This project aims to develop a machine learning model to predict flight ticket prices based on multiple factors such as the airline, travel date, source, destination, duration, and number of stops. By accurately predicting ticket prices, the model will:
- **Assist Consumers**: Help travelers identify the best time to book flights at the most affordable price.
- **Optimize Airline Pricing**: Provide airlines and travel platforms with insights into market trends and pricing optimization strategies.

## Data
### Dataset Column Descriptions
Below is the description of each column in the dataset to help users understand its contents:
1. **`Unnamed: 0`**: An automatically generated index column with no direct relevance to analysis or prediction.
2. **`airline`**: The name of the airline offering the ticket. Examples include "SpiceJet," "AirAsia," and "Vistara."
3. **`flight`**: A unique identifier for each flight, usually represented by flight codes such as "SG-8709."
4. **`source_city`**: The departure city of the flight, indicating the starting point of the passenger's journey. Example: "Delhi."
5. **`departure_time`**: The departure time of the flight, grouped into periods such as "Morning," "Afternoon," "Evening," and "Night."
6. **`stops`**: The number of stops during the journey. Examples: "zero" (direct flight), "one" (one stop), or "two_or_more" (two or more stops).
7. **`arrival_time`**: The arrival time of the flight, also grouped into time periods like "Morning," "Afternoon," "Evening," and "Night."
8. **`destination_city`**: The destination city of the flight, representing the end point of the passenger's journey. Example: "Mumbai."
9. **`class`**: The travel class of the ticket, such as "Economy" or "Business."
10. **`duration`**: The duration of the journey in hours, including flight time and stopovers (if any). Example: 2.17 hours.
11. **`days_left`**: The number of days left until the departure date from the time the ticket was purchased.
12. **`price`**: The price of the flight ticket in Indian Rupee (INR) currency. This is the target variable to be predicted by the machine learning model.

### Usage Example
With this description, dataset users can understand that features like **`airline`**, **`class`**, **`days_left`**, and **`duration`** are key factors influencing **`price`**, the target variable.

## Method
The project will focus solely on predicting flight ticket prices using the dataset attributes and providing actionable insights for consumers and businesses. We aim to achieve accurate predictions by leveraging regression-based machine learning models. The performance of the models will be evaluated using metrics such as Mean Squared Error (MSE) and R². The best model will be selected based on cross-validation results, specifically focusing on the mean MSE and its standard deviation, to ensure both accuracy and consistency in predictions.

## Stacks
### Data manipulation and analysis
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

### Visualization
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization

### Model selection and evaluation
- `scikit-learn` - Model selection, evaluation, and machine learning algorithms

### Preprocessing
- `OneHotEncoder` - Encoding categorical features as a one-hot numeric array
- `OrdinalEncoder` - Encoding categorical features as an ordinal array
- `StandardScaler` - Standardizing features by removing the mean and scaling to unit variance
- `ColumnTransformer` - Applying different preprocessing steps to different subsets of features
- `Pipeline` - Assembling several steps that can be cross-validated together

### Models
- `LinearRegression` - Linear regression model
- `SVR` - Support Vector Regression
- `RandomForestRegressor` - Random Forest Regressor
- `GradientBoostingRegressor` - Gradient Boosting Regressor
- `KNeighborsRegressor` - K-Nearest Neighbors Regressor
- `DecisionTreeRegressor` - Decision Tree Regressor

### Utilities
- `joblib` - Serialization and deserialization of Python objects
- `scipy.stats` - Statistical functions and distributions
     - `randint`
     - `uniform`

## Target Users
- **Travel Agencies and Platforms**: To provide better price recommendations for travelers.
- **Airlines**: To optimize pricing strategies based on market trends and demand.
- **Travelers**: To make informed decisions on when and where to book flights for cost savings.
