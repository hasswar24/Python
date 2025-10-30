import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data from downloaded file
oil_price_data = pd.read_csv("/Users/hassaan/Documents/Nat_Gas.csv")


oil_price_data['Dates'] = pd.to_datetime(oil_price_data["Dates"])
oil_price_data['Days'] = (oil_price_data['Dates'] - oil_price_data['Dates'].min()).dt.days     #calculates the days between the beginning of the data set and 

# Get user input date and convert it to datetime
SpecificDate = pd.to_datetime(input("Enter the date to predict the oil price (yyyy-mm-dd): "))

# Calculate the number of days from the start of the dataset to the SpecificDate
days_from_start = (SpecificDate - oil_price_data['Dates'].min()).days


if days_from_start <= oil_price_data['Days'].max():
    print("Error: The date entered must be in the future relative to the latest date in the dataset.")  #
else:
    # Prepare features and target for the model
    Days = oil_price_data[['Days']]  # Features (independent variable)
    Prices = oil_price_data['Prices']  # Target (dependent variable)

    # Train linear regression model
    model = LinearRegression()
    model.fit(Days, Prices)

    # Predict the oil price for the SpecificDate
    future_user_day = np.array([[days_from_start]])
    future_price_pred = model.predict(future_user_day)


    # Output the predicted price for the user-specified date
    print(f"Predicted oil price for {SpecificDate} is: ${future_price_pred[0]:.2f}")    #outputs the first value of the future price prediction. The future price is predicted using the first value from the array and outputted as 2 dp.

  
    future_days = np.array(range(Days['Days'].max() + 1, Days['Days'].max() + 365*3)).reshape(-1, 1)   #used for making the plot of future prices. 365*3 represents making price predicitons for the next 3 years. + 1 added to days to ensure prediction starts from beyond data set
    predictions = model.predict(future_days)

    # Plot historical and predicted prices with the specific date prediction

    plt.plot(oil_price_data['Dates'], oil_price_data['Prices'], label='Historical Prices', color='blue') #plots data in file
    plt.plot(oil_price_data['Dates'].min() + pd.to_timedelta(future_days.flatten(), unit='D'), predictions, 
             label='Predicted Prices', color='green', marker='o', markersize=1)
    
    # Highlight user-specified prediction
    plt.scatter(SpecificDate, future_price_pred, color='red', label='Prediction for Specific Date', zorder=5)

    # Plot details
    plt.title("Oil Prices over Time")
    plt.xlabel("Date")
    plt.ylabel("Prices")
    plt.xticks(rotation=270) #rotates x axis labels for better presentation 
    plt.legend()
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
