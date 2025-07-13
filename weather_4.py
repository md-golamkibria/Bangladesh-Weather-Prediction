import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# Disable XGBoost usage if libomp is not available to avoid runtime errors
import os
import platform

if platform.system() == 'Darwin':
    # macOS detected, check if libomp is installed
    import ctypes.util
    libomp_path = ctypes.util.find_library('omp')
    if libomp_path is None:
        # Disable XGBoost usage due to missing libomp
        XGBRegressor = None

# Additional fallback: disable XGBoost if import succeeded but runtime error occurs
# We will wrap XGBoost model training and prediction in try-except blocks to catch runtime errors

from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

import warnings

import matplotlib.pyplot as plt

import calendar

import random

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense

from tensorflow.keras.callbacks import EarlyStopping



warnings.filterwarnings("ignore")



class AIWeatherPrediction:

    def __init__(self, data_path):

        self.data_path = data_path

        self.data = None

        self.models = {}

        self.results = {}

        # Define thresholds for classification

        self.temp_thresholds = {

            'very_warm': 30,  # degrees Celsius

            'normal_min': 15,

            'normal_max': 30

        }

        self.rain_thresholds = {

            'rainy': 50  # mm of rain per month

        }

        # For LSTM data

        self.lstm_seq_length = 3  # number of months in sequence



    def load_and_preprocess(self):

        # Load data

        self.data = pd.read_csv(self.data_path)

        # Check for missing values and drop if any

        self.data.dropna(inplace=True)

        # Features and targets

        self.X = self.data[['Year', 'Month']]

        self.y_temp = self.data['tem']

        self.y_rain = self.data['rain']



    def create_lstm_sequences(self, X, y_temp, y_rain):

        """

        Create sequences of data for LSTM input.

        Each sequence contains lstm_seq_length months of features.

        """

        seq_length = self.lstm_seq_length

        X_seq = []

        y_temp_seq = []

        y_rain_seq = []

        data_len = len(X)

        for i in range(data_len - seq_length):

            X_seq.append(X.iloc[i:i+seq_length].values)

            y_temp_seq.append(y_temp.iloc[i+seq_length])

            y_rain_seq.append(y_rain.iloc[i+seq_length])

        return np.array(X_seq), np.array(y_temp_seq), np.array(y_rain_seq)



    def build_lstm_model(self):

        model = Sequential()

        model.add(LSTM(50, activation='relu', input_shape=(self.lstm_seq_length, 2)))

        model.add(Dense(25, activation='relu'))

        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        return model



    def train_models(self):

        # Split data into train and test sets

        X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(

            self.X, self.y_temp, self.y_rain, test_size=0.2, random_state=42)



        self.X_test = X_test

        self.y_temp_test = y_temp_test

        self.y_rain_test = y_rain_test



        # Train classical models as before

        dt_temp = DecisionTreeRegressor(random_state=42)

        dt_temp.fit(X_train, y_temp_train)

        dt_rain = DecisionTreeRegressor(random_state=42)

        dt_rain.fit(X_train, y_rain_train)



        rf_temp = RandomForestRegressor(random_state=42)

        rf_temp.fit(X_train, y_temp_train)

        rf_rain = RandomForestRegressor(random_state=42)

        rf_rain.fit(X_train, y_rain_train)



        knn_temp = KNeighborsRegressor()

        knn_temp.fit(X_train, y_temp_train)

        knn_rain = KNeighborsRegressor()

        knn_rain.fit(X_train, y_rain_train)



        gb_temp = GradientBoostingRegressor(random_state=42)

        gb_temp.fit(X_train, y_temp_train)

        gb_rain = GradientBoostingRegressor(random_state=42)

        gb_rain.fit(X_train, y_rain_train)



        svr_temp = SVR()

        svr_temp.fit(X_train, y_temp_train)

        svr_rain = SVR()

        svr_rain.fit(X_train, y_rain_train)

        # Train XGBoost model if available

        if XGBRegressor is not None:

            xgb_temp = XGBRegressor(random_state=42, verbosity=0)

            xgb_temp.fit(X_train, y_temp_train)

            xgb_rain = XGBRegressor(random_state=42, verbosity=0)

            xgb_rain.fit(X_train, y_rain_train)

        else:

            xgb_temp = None

            xgb_rain = None



        temp_bins = np.linspace(self.y_temp.min(), self.y_temp.max(), 20)

        y_temp_train_binned = np.digitize(y_temp_train, temp_bins)

        y_temp_test_binned = np.digitize(y_temp_test, temp_bins)

        gnb_temp = GaussianNB()

        gnb_temp.fit(X_train, y_temp_train_binned)




        rain_bins = np.linspace(self.y_rain.min(), self.y_rain.max(), 20)

        y_rain_train_binned = np.digitize(y_rain_train, rain_bins)

        y_rain_test_binned = np.digitize(y_rain_test, rain_bins)

        gnb_rain = GaussianNB()

        gnb_rain.fit(X_train, y_rain_train_binned)



        # Train LSTM model

        X_train_seq, y_temp_train_seq, y_rain_train_seq = self.create_lstm_sequences(X_train.reset_index(drop=True),

                                                                                   y_temp_train.reset_index(drop=True),

                                                                                   y_rain_train.reset_index(drop=True))

        X_test_seq, y_temp_test_seq, y_rain_test_seq = self.create_lstm_sequences(X_test.reset_index(drop=True),

                                                                                 y_temp_test.reset_index(drop=True),

                                                                                 y_rain_test.reset_index(drop=True))



        self.X_test_seq = X_test_seq

        self.y_temp_test_seq = y_temp_test_seq

        self.y_rain_test_seq = y_rain_test_seq



        lstm_temp_model = self.build_lstm_model()

        lstm_temp_model.fit(X_train_seq, y_temp_train_seq, epochs=50, batch_size=16, verbose=0,

                            validation_data=(X_test_seq, y_temp_test_seq),

                            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])



        lstm_rain_model = self.build_lstm_model()

        lstm_rain_model.fit(X_train_seq, y_rain_train_seq, epochs=50, batch_size=16, verbose=0,

                           validation_data=(X_test_seq, y_rain_test_seq),

                           callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])



        self.models = {

            'Decision Tree': {'temp': dt_temp, 'rain': dt_rain},

            'Random Forest': {'temp': rf_temp, 'rain': rf_rain},

            'KNN': {'temp': knn_temp, 'rain': knn_rain},

            'Gradient Boosting': {'temp': gb_temp, 'rain': gb_rain},

            'SVM': {'temp': svr_temp, 'rain': svr_rain},

            'Naive Bayes': {'temp': gnb_temp, 'rain': gnb_rain,

                            'temp_bins': temp_bins, 'rain_bins': rain_bins},

            'LSTM': {'temp': lstm_temp_model, 'rain': lstm_rain_model},

            'XGBoost': {'temp': xgb_temp, 'rain': xgb_rain}

        }



    def evaluate_models(self):

        report = {}

        for model_name, model_dict in self.models.items():

            # Skip models that are None (e.g., XGBoost if not trained)

            if model_dict['temp'] is None or model_dict['rain'] is None:

                continue

            if model_name == 'Naive Bayes':

                temp_pred_bins = model_dict['temp'].predict(self.X_test)

                rain_pred_bins = model_dict['rain'].predict(self.X_test)



                temp_bin_centers = (model_dict['temp_bins'][:-1] + model_dict['temp_bins'][1:]) / 2

                rain_bin_centers = (model_dict['rain_bins'][:-1] + model_dict['rain_bins'][1:]) / 2



                temp_pred = temp_bin_centers[temp_pred_bins - 1]

                rain_pred = rain_bin_centers[rain_pred_bins - 1]

                temp_true = np.digitize(self.y_temp_test, model_dict['temp_bins'])

                rain_true = np.digitize(self.y_rain_test, model_dict['rain_bins'])

                temp_f1 = f1_score(temp_true, temp_pred_bins, average='weighted')

                rain_f1 = f1_score(rain_true, rain_pred_bins, average='weighted')

            elif model_name == 'LSTM':

                temp_pred = model_dict['temp'].predict(self.X_test_seq).flatten()

                rain_pred = model_dict['rain'].predict(self.X_test_seq).flatten()

                temp_true = self.y_temp_test_seq

                rain_true = self.y_rain_test_seq

            else:

                temp_pred = model_dict['temp'].predict(self.X_test)

                rain_pred = model_dict['rain'].predict(self.X_test)

                temp_true = self.y_temp_test

                rain_true = self.y_rain_test



            temp_mae = mean_absolute_error(temp_true, temp_pred)

            temp_rmse = np.sqrt(mean_squared_error(temp_true, temp_pred))

            rain_mae = mean_absolute_error(rain_true, rain_pred)

            rain_rmse = np.sqrt(mean_squared_error(rain_true, rain_pred))



            if model_name == 'Naive Bayes':

                report[model_name] = {

                    'Temperature MAE': temp_mae,

                    'Temperature RMSE': temp_rmse,

                    'Rain MAE': rain_mae,

                    'Rain RMSE': rain_rmse,

                    'Temperature F1 Score': temp_f1,

                    'Rain F1 Score': rain_f1

                }

            else:

                report[model_name] = {

                    'Temperature MAE': temp_mae,

                    'Temperature RMSE': temp_rmse,

                    'Rain MAE': rain_mae,

                    'Rain RMSE': rain_rmse

                }

        self.results = report



    def predict(self, year, month):

        def get_next_month(year, month):

            if month == 12:

                return year + 1, 1

            else:

                return year, month + 1



        input_df = pd.DataFrame({'Year': [year], 'Month': [month]})

        next_year, next_month = get_next_month(year, month)

        next_input_df = pd.DataFrame({'Year': [next_year], 'Month': [next_month]})



        predictions = {}

        for model_name, model_dict in self.models.items():

            # Skip models that are None (e.g., XGBoost if not trained)

            if model_dict['temp'] is None or model_dict['rain'] is None:

                continue

            if model_name == 'Naive Bayes':

                temp_pred_bin = model_dict['temp'].predict(input_df)[0]

                rain_pred_bin = model_dict['rain'].predict(input_df)[0]



                temp_bin_centers = (model_dict['temp_bins'][:-1] + model_dict['temp_bins'][1:]) / 2

                rain_bin_centers = (model_dict['rain_bins'][:-1] + model_dict['rain_bins'][1:]) / 2



                temp_pred = temp_bin_centers[temp_pred_bin - 1]

                rain_pred = rain_bin_centers[rain_pred_bin - 1]



                next_temp_pred_bin = model_dict['temp'].predict(next_input_df)[0]

                next_rain_pred_bin = model_dict['rain'].predict(next_input_df)[0]



                next_temp_pred = temp_bin_centers[next_temp_pred_bin - 1]

                next_rain_pred = rain_bin_centers[next_rain_pred_bin - 1]

            elif model_name == 'LSTM':

                # For LSTM, create sequence input for prediction

                # We need to create a sequence of length lstm_seq_length ending with the input year and month

                # For simplicity, use the last lstm_seq_length-1 months from training data plus current input

                # This is a simplification; ideally, we would have a time series to generate sequences properly

                # Here, we will create a sequence with repeated input for demonstration

                seq_length = self.lstm_seq_length

                input_seq = np.array([[year, month]] * seq_length).reshape(1, seq_length, 2)

                next_year, next_month = get_next_month(year, month)

                next_input_seq = np.array([[next_year, next_month]] * seq_length).reshape(1, seq_length, 2)



                temp_pred = model_dict['temp'].predict(input_seq)[0][0]

                rain_pred = model_dict['rain'].predict(input_seq)[0][0]

                next_temp_pred = model_dict['temp'].predict(next_input_seq)[0][0]

                next_rain_pred = model_dict['rain'].predict(next_input_seq)[0][0]

            else:

                temp_pred = model_dict['temp'].predict(input_df)[0]

                rain_pred = model_dict['rain'].predict(input_df)[0]



                next_temp_pred = model_dict['temp'].predict(next_input_df)[0]

                next_rain_pred = model_dict['rain'].predict(next_input_df)[0]



            predictions[model_name] = {

                'Current Month': {'Temperature': temp_pred, 'Rain': rain_pred},

                'Next Month': {'Temperature': next_temp_pred, 'Rain': next_rain_pred}

            }

        return predictions



    def print_report(self):

        print("Model Performance Report:")

        for model_name, metrics in self.results.items():

            print(f"\n{model_name}:")

            for metric_name, value in metrics.items():

                if 'F1 Score' in metric_name:

                    print(f"  {metric_name}: {value:.4f}")

                else:

                    print(f"  {metric_name}: {value:.4f}")





    def generate_yearly_report(self, year):

        months = list(range(1, 13))

        temp_preds = {model: [] for model in self.models.keys()}

        rain_preds = {model: [] for model in self.models.keys()}



        for month in months:

            preds = self.predict(year, month)

            for model_name, pred in preds.items():

                temp_preds[model_name].append(pred['Current Month']['Temperature'])

                rain_preds[model_name].append(pred['Current Month']['Rain'])



        # Remove models with empty predictions (e.g., XGBoost if not trained)

        temp_preds = {k: v for k, v in temp_preds.items() if len(v) == 12}

        rain_preds = {k: v for k, v in rain_preds.items() if len(v) == 12}



        # Plot Temperature

        plt.figure(figsize=(12, 6))

        for model_name, temps in temp_preds.items():

            plt.plot(months, temps, label=model_name)

        plt.title(f'Temperature Predictions for Year {year}')

        plt.xlabel('Month')

        plt.ylabel('Temperature')

        plt.legend()

        plt.grid(True)

        plt.xticks(months)

        plt.show()



        # Plot Rain

        plt.figure(figsize=(12, 6))

        for model_name, rains in rain_preds.items():

            plt.plot(months, rains, label=model_name)

        plt.title(f'Rain Predictions for Year {year}')

        plt.xlabel('Month')

        plt.ylabel('Rain')

        plt.legend()

        plt.grid(True)

        plt.xticks(months)

        plt.show()



    def classify_month(self, temperature, rain):

        """

        Classify the month based on temperature and rain thresholds.

        Returns a dict with classification for temperature and rain.

        """

        temp_class = 'normal'

        rain_class = 'normal'



        if temperature >= self.temp_thresholds['very_warm']:

            temp_class = 'very warm'

        elif self.temp_thresholds['normal_min'] <= temperature < self.temp_thresholds['normal_max']:

            temp_class = 'normal'

        else:

            temp_class = 'cold'



        if rain >= self.rain_thresholds['rainy']:

            rain_class = 'rainy'

        else:

            rain_class = 'dry'



        return {'temperature': temp_class, 'rain': rain_class}



    def simulate_daily_weather(self, year, month, temperature, rain):

        """

        Simulate daily weather conditions for the given month based on monthly temperature and rain.

        Returns a dict with dates as keys and weather classification as values.

        For rainy days, also includes probable rain start and end times in 12-hour AM/PM format.

        """

        days_in_month = calendar.monthrange(year, month)[1]



        daily_weather = {}



        # Simple heuristic: distribute rain days proportional to rain amount

        # Assume if rain > threshold, some days are rainy, else dry

        rain_days_count = int(min(rain / 10, days_in_month))  # heuristic: 1 rain day per 10 mm rain



        # Assign rain days randomly

        rain_days = set(random.sample(range(1, days_in_month + 1), rain_days_count)) if rain_days_count > 0 else set()



        for day in range(1, days_in_month + 1):

            if day in rain_days:

                # Assign random probable rain start and end times between 6 AM and 9 PM

                start_hour = random.randint(6, 18)  # start between 6 AM and 6 PM

                end_hour = random.randint(start_hour + 1, 21)  # end at least 1 hour after start, max 9 PM

                start_minute = random.choice([0, 15, 30, 45])

                end_minute = random.choice([0, 15, 30, 45])



                start_am_pm = 'AM' if start_hour < 12 else 'PM'

                end_am_pm = 'AM' if end_hour < 12 else 'PM'



                start_hour_12 = start_hour if 1 <= start_hour <= 12 else start_hour - 12 if start_hour > 12 else 12

                end_hour_12 = end_hour if 1 <= end_hour <= 12 else end_hour - 12 if end_hour > 12 else 12



                start_time_str = f"{start_hour_12}:{start_minute:02d} {start_am_pm}"

                end_time_str = f"{end_hour_12}:{end_minute:02d} {end_am_pm}"



                weather = f"rainy from {start_time_str} to {end_time_str}"

            else:

                # Classify temperature for the day

                if temperature >= self.temp_thresholds['very_warm']:

                    weather = 'very warm'

                elif self.temp_thresholds['normal_min'] <= temperature < self.temp_thresholds['normal_max']:

                    weather = 'normal'

                else:

                    weather = 'cold'

            daily_weather[day] = weather



        return daily_weather



    def predict_and_classify_daily(self, year, month):

        """

        Predict temperature and rain for the month and simulate daily weather classification.

        Returns a dict with model names as keys and daily weather dicts as values.

        """

        predictions = self.predict(year, month)

        daily_classifications = {}

        for model_name, pred in predictions.items():

            temp = pred['Current Month']['Temperature']

            rain = pred['Current Month']['Rain']

            daily_weather = self.simulate_daily_weather(year, month, temp, rain)

            daily_classifications[model_name] = daily_weather

        return daily_classifications



    def main():

        data_path = '../Downloads/archive/sorted_temp_and_rain_dataset.csv'

        weather_ai = AIWeatherPrediction(data_path)

        weather_ai.load_and_preprocess()

        weather_ai.train_models()

        weather_ai.evaluate_models()

        weather_ai.print_report()



        # Example input

        year = int(input("Enter year: "))

        month = int(input("Enter month (1-12): "))



        predictions = weather_ai.predict(year, month)

        print(f"\nPredictions for Year: {year}, Month: {month} and Next Month")

        for model_name, preds in predictions.items():

            print(f"{model_name} -> Current Month -> Temperature: {preds['Current Month']['Temperature']:.2f}, Rain: {preds['Current Month']['Rain']:.2f}")

            print(f"{model_name} -> Next Month -> Temperature: {preds['Next Month']['Temperature']:.2f}, Rain: {preds['Next Month']['Rain']:.2f}")

            if model_name == 'Naive Bayes':

                # Calculate and show F1 score for the input month

                temp_true_bin = weather_ai.models['Naive Bayes']['temp'].classes_

                rain_true_bin = weather_ai.models['Naive Bayes']['rain'].classes_

                import numpy as np

                from sklearn.metrics import f1_score

                # Digitize true values for the input month

                temp_true = np.digitize([weather_ai.y_temp_test.iloc[0]], weather_ai.models['Naive Bayes']['temp_bins'])

                rain_true = np.digitize([weather_ai.y_rain_test.iloc[0]], weather_ai.models['Naive Bayes']['rain_bins'])

                temp_pred_bin = weather_ai.models['Naive Bayes']['temp'].predict(pd.DataFrame({'Year': [year], 'Month': [month]}))

                rain_pred_bin = weather_ai.models['Naive Bayes']['rain'].predict(pd.DataFrame({'Year': [year], 'Month': [month]}))

                temp_f1 = f1_score(temp_true, temp_pred_bin, average='weighted')

                rain_f1 = f1_score(rain_true, rain_pred_bin, average='weighted')

                print(f"  {model_name} -> Temperature F1 Score: {temp_f1:.4f}")

                print(f"  {model_name} -> Rain F1 Score: {rain_f1:.4f}")



        # Generate yearly report graphs

        weather_ai.generate_yearly_report(year)



        # Show daily classification for the input month using Random Forest model

        daily_classification = weather_ai.predict_and_classify_daily(year, month)

        print(f"\nDaily Weather Classification for Year: {year}, Month: {month} (Random Forest Model):")

        for day, weather in daily_classification['Random Forest'].items():

            print(f"Day {day}: {weather}")



def main():
    data_path = '../Downloads/archive/sorted_temp_and_rain_dataset.csv'
    weather_ai = AIWeatherPrediction(data_path)
    weather_ai.load_and_preprocess()
    weather_ai.train_models()
    weather_ai.evaluate_models()
    weather_ai.print_report()

    # Example input
    year = int(input("Enter year: "))
    month = int(input("Enter month (1-12): "))

    predictions = weather_ai.predict(year, month)

    print(f"\nPredictions for Year: {year}, Month: {month} and Next Month")

    # Calculate average temperature and rain for current and next month across all models

    avg_temp_current = 0

    avg_rain_current = 0

    avg_temp_next = 0

    avg_rain_next = 0

    model_count = len(predictions)

    for model_name, preds in predictions.items():

        avg_temp_current += preds['Current Month']['Temperature']

        avg_rain_current += preds['Current Month']['Rain']

        avg_temp_next += preds['Next Month']['Temperature']

        avg_rain_next += preds['Next Month']['Rain']

    avg_temp_current /= model_count

    avg_rain_current /= model_count

    avg_temp_next /= model_count

    avg_rain_next /= model_count

    print(f"Average -> Current Month -> Temperature: {avg_temp_current:.2f}, Rain: {avg_rain_current:.2f}")

    print(f"Average -> Next Month -> Temperature: {avg_temp_next:.2f}, Rain: {avg_rain_next:.2f}")



    for model_name, preds in predictions.items():

        print(f"{model_name} -> Current Month -> Temperature: {preds['Current Month']['Temperature']:.2f}, Rain: {preds['Current Month']['Rain']:.2f}")

        print(f"{model_name} -> Next Month -> Temperature: {preds['Next Month']['Temperature']:.2f}, Rain: {preds['Next Month']['Rain']:.2f}")

        if model_name == 'Naive Bayes':

            import numpy as np

            from sklearn.metrics import f1_score

            temp_true = np.digitize([weather_ai.y_temp_test.iloc[0]], weather_ai.models['Naive Bayes']['temp_bins'])

            rain_true = np.digitize([weather_ai.y_rain_test.iloc[0]], weather_ai.models['Naive Bayes']['rain_bins'])

            temp_pred_bin = weather_ai.models['Naive Bayes']['temp'].predict(pd.DataFrame({'Year': [year], 'Month': [month]}))

            rain_pred_bin = weather_ai.models['Naive Bayes']['rain'].predict(pd.DataFrame({'Year': [year], 'Month': [month]}))

            temp_f1 = f1_score(temp_true, temp_pred_bin, average='weighted')

            rain_f1 = f1_score(rain_true, rain_pred_bin, average='weighted')

            print(f"  {model_name} -> Temperature F1 Score: {temp_f1:.4f}")

            print(f"  {model_name} -> Rain F1 Score: {rain_f1:.4f}")

    # Generate yearly report graphs
    weather_ai.generate_yearly_report(year)

    # Show daily classification for the input month using Random Forest model
    # Replace with classification based on average temperature and rain
    daily_weather = weather_ai.simulate_daily_weather(year, month, avg_temp_current, avg_rain_current)
    print(f"\nDaily Weather Classification for Year: {year}, Month: {month} (Based on Average Predictions):")
    for day, weather in daily_weather.items():
        print(f"Day {day}: {weather}")

if __name__ == "__main__":
    main()