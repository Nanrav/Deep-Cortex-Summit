import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go

# Function to predict stock prices and calculate metrics
def predict_stock(data, column):
    dataset = data[[column]].values.astype('float32')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split into train and test sets
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Create dataset function
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Prepare data for LSTM
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))  # Adjusted LSTM units for better learning
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate RMSE
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    # Calculate accuracy percentage based on a margin
    error_margin = 0.05  # 5% error margin
    accuracy = np.mean(np.abs((testPredict[:, 0] - testY[0]) / testY[0]) < error_margin) * 100

    # Prepare data for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    original_data = scaler.inverse_transform(dataset)

    # Create interactive plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(original_data)), y=original_data[:, 0],
                             mode='lines', name='Actual Close', line=dict(color='deepskyblue')))
    fig.add_trace(go.Scatter(x=np.arange(len(trainPredictPlot)), y=trainPredictPlot[:, 0],
                             mode='lines', name='Training Predictions', line=dict(dash='dash', color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(testPredictPlot)), y=testPredictPlot[:, 0],
                             mode='lines', name='Test Predictions', line=dict(dash='dash', color='red')))

    # Customize plot layout to make it bigger and center it
    fig.update_layout(
        title=f'LSTM STOCK PRICE PREDICTION',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        width=1000,  # Increased width
        height=600,  # Increased height
        title_font=dict(size=24, color='white', family='Arial', weight='bold')  # Make the title bold and visible
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig, trainScore, testScore, accuracy, testPredict[:, 0], data.index[-len(testPredict):]  # Return all required values

# Streamlit UI
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS for dark theme
st.markdown("""<style>
body {
    background-color: #1E1E1E;
    color: white;
}
.css-1d391kg { 
    background-color: #333333; 
}
h1, h2, h3, h4, h5, h6 {
    color: white;
}
.stSelectbox label, .stFileUploader label {
    color: white;
}
</style>""", unsafe_allow_html=True)

# Add title bar
st.title("LSTM Stock Price Prediction")

# Sidebar for stock data input
st.sidebar.title("Stock Data Input")

# Use yfinance to download data dynamically
stock_symbol = st.sidebar.text_input("Enter stock symbol (e.g., AAPL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

if st.sidebar.button("Predict"):
    # Fetch data from yfinance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        st.error("No data found for the given stock symbol and date range.")
    else:
        # Use 'Close' price for prediction
        column_to_predict = 'Close'

        with st.spinner('Predicting...'):
            fig, train_score, test_score, accuracy, predictions, predicted_dates = predict_stock(data, column_to_predict)

        # Center the graph and make it bigger
        st.plotly_chart(fig, use_container_width=True)

        # Display RMSE scores
        st.write(f"Training RMSE: {train_score:.2f}")
        st.write(f"Test RMSE: {test_score:.2f}")
        st.write(f"Model Accuracy: {accuracy:.2f}%")  # Display the accuracy

        # Display numerical predicted values for the test set
        st.subheader("Predicted Test Set Values:")
        st.write(predictions)

# Chatbot feature
st.sidebar.title("Chatbot")

user_input = st.sidebar.text_input("Ask a question about the predictions:", "")

if user_input:
    response = ""
    user_input_lower = user_input.lower()

    if "accuracy" in user_input_lower:
        response = f"The model accuracy is {accuracy:.2f}%."
    elif "rmse" in user_input_lower:
        response = f"Training RMSE: {train_score:.2f}, Test RMSE: {test_score:.2f}."
    elif "predictions" in user_input_lower:
        response = "The predictions are the expected future stock prices based on the model."
    elif "predicted value" in user_input_lower:
        # Extract the date from the user's input
        try:
            date_str = user_input_lower.split("on")[-1].strip()  # Extract text after "on"
            date = pd.to_datetime(date_str)

            # Find the index of the predicted date
            if date in predicted_dates:
                index = list(predicted_dates).index(date)
                predicted_value = predictions[index]
                response = f"The predicted value for {date.strftime('%Y-%m-%d')} is ${predicted_value:.2f}."
            else:
                response = "Sorry, I don't have a prediction for that date."
        except Exception as e:
            response = "Please specify a valid date in your question (e.g., 'What is the predicted value on YYYY-MM-DD')."

    else:
        response = "I'm sorry, I can only provide information about the accuracy, RMSE, predictions, and predicted values for specific dates."

    st.sidebar.write(f"**Chatbot Response:** {response}")
