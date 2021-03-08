---
layout: post
title: Regression Based Prediction for Precipitation using Global Climate Data
subtitle: This project was to use Multiple Linear Regression (MLR) Machine Learning models on Global Climate Data from NASA to make precipitation prediction.
cover-img: /assets/img/Precipitation-Cover.jpg
thumbnail-img: /assets/img/Precipitation-Cover.jpg
share-img: /assets/img/Precipitation-Cover.jpg
tags: [python, MachineLearning, Precipitation, Prediction, model]
---
## Abstract

 Climate is becoming increasingly unpredictable over decades and it has never been more critical to make a better prediction on climate in human history. There is a substantial human and capital cost involved with a severe climate event. Current meteorological models can predict the climate in any area of the world with great accuracy. This research aims at predicting Precipitation using historical information and leveraging Multiple Linear Regression (MLR) in python Machine Learning to make a prediction. The prediction does not have to be for the following week or month, but it can be many years in the future. Multiple models were run using weather data from NASA and at best, an R^2 of 48.5% was obtained. The R^2 value may seem low but the Mean Absolute Error (MAE) has a 67% improvement from the baseline. The low R^2 was attributed to the non-availability of crucial climate information that could be added later to make a more refined prediction. Temperature was identified as the key feature in predicting Precipitation. This research's best model's predictions may enable decision-makers like governments or even insurance companies (with financial interests) to make predictions to plan for policies or undertaken risks.

## Finding the Data and the JSON hurdle

Finding the data itself was a big hurdle in this project. The data must provide a learning opportunity and at the same time, Machine Learning practices can be applied on it. The data must have enough observations to devote time to. At least 100,000 so that there are enough observations to train, validate and then test. On the other hand, the data must relate to something that can tie into a business case.

After much investigation, it was finalized to take up the topic first to predict Rainfall. Precipitation was the only thing that was closest to the topic of interest. 

NASA maintains an app called POWER Single Point Data Access <sup>1</sup> that provides data in JavaScript Object Notation (JSON) format through an Application Programming Interface (API) to users based on the geographical location of interest. NASA's POWER app requires latitude and longitude to provide the weather information. The latitude and longitude of various countries were gathered using an existing CSV file from GitHub user albertyw <sup>2</sup>.

A program was written in python notebook that takes in the latitude longitude information from the country list, pass it to NASA's app to fetch the following information:-

1. Lattitude (degrees in decimal)
2. Longitude (degrees in decimal)
3. Elevation (m)
4. PRECTOT - Precipitation (mm day-1)
5. QV2M - Specific Humidity at 2 Meters (g/kg)
6. PS - Surface Pressure (kPa)
7. TS - Earth Skin Temperature (C)
8. T2MDEW - Dew/Frost Point at 2 Meters (C)
9. T2M - Temperature Range at 2 Meters (C)
10. WS50M - Wind Speed at 50 Meters (m/s)
11. WS10M - Wind Speed at 10 Meters (m/s)
12. T2MWET - Wet Bulb Temperature at 2 Meters (C)
13. T2M_RANGE - Temperature Range at 2 Meters (C)
14. RH2M - Relative Humidity at 2 Meters (%)
15. KT - Insolation Clearness Index (dimensionless)
16. CLRSKY_SFC_SW_DWN - Clear Sky Insolation Incident on a Horizontal Surface (kW-hr/m^2/day)
17. ALLSKY_SFC_SW_DWN - All Sky Insolation Incident on a Horizontal Surface (kW-hr/m^2/day)
18. ALLSKY_SFC_LW_DWN - Downward Thermal Infrared (Longwave) Radiative Flux (kW-hr/m^2/day)

The python code also merges the country code, latitude and longitude data to make a single data frame for use. The data included weather data for each day for 240 countries from 1980 to 2020. The resulting dataset had 3506400 rows × 22 columns.

![NASA-POWER-DataFrame](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/NASA-POWER-DataFrame.png)
*NASA-POWER-DataFrame*

The data frame built was used in the research further. The learning to now be able to use JSON data available publically, send JSON requests, receive and interpret and convert them to pandas DataFrame was a small achievement in the machine learning model.

---
## EDA and Machine Learning Model

Data from NASA's application was cleaned and unnecessary or repetitive columns were dropped. Before getting into features and target selection, another feature was included in the data that should affect the Precipitation of any interest area. Based on a hypothesis that Precipitation will be higher in countries with more forest areas, Forest Area data from World Bank <sup>3</sup> was imported in the data frame as a feature through pandas merge function.

Due to resource limitations and quick turnarounds in model training, only the last twenty years of data were considered.

The data was ready for some Exploratory Data Analysis (EDA). Pandas Profiling was used to generate a report to see the type of data, missing values and data distribution.

The features were selected to be the following:-

1. country_code
2. lat
3. long
4. elevation
5. surface_pressure
6. skin_temperature
7. dew_frost
8. temperature2m
9. windspeed10m
10. windspeed50m
11. wet_bulb_temp
12. temp_range
13. clearness_index
14. clear_sky_insolation
15. all_sky_insolation
16. radiative_flux
17. Forest_Cover(sq km)

The definition of all the features mentioned above was provided in the text above. **Precipitation** was chosen as the **target**.

The target was skewed to the right due to the presence of some 300 observations.
### Baseline

 Precipitation mean was chosen to set a baseline to compare the model performance. Precipitation mean was calculated for the entire data and was determined as **2.787 mm**. Mean Absolute Error (MAE) was calculated and was found to be **3.379 mm**. The baseline MAE is used to compare various models to see how each model fare in precipitation predictions.

### Models

The model pipelines that use more time and resources in fitting the data frame from 2001 - 2020 was split as follows:-

Train - Data from 2008 - 2012
Validate - Data from 2013 only
Test - Data from 2014

The various models run and their findings are discussed below:-

#### 1. Ordinal Encoder and RandomForestRegressor pipeline

The code to instantiate and fit the pipeline was as simple as:-

{% highlight python linenos %}
pipeline_randomforest_OE = make_pipeline(
    ce.OrdinalEncoder(),
    RandomForestRegressor(n_estimators=100, random_state=42, verbose=1,n_jobs=-1)
)
pipeline_randomforest_OE.fit(X_train, y_train)
{% endhighlight %}

Since the data was super clean with no missing values, compute or scaling were not used.

Parameters to benchmark the model:-

| Parameter | Value |
| :------ |:--- |
| Time to fit the model | 22 sec |
| Training Score (R<sup>2</sup>) | 93.70 % |
| Validation Score (R<sup>2</sup>) | 44.02 % |
| Test Score (R<sup>2</sup>) | 44.22 % |
| Baseline MAE | 3.379 mm
| Model MAE | 2.198 mm
| Improvement over Baseline MAE | 53.73 %

#### 2. OneHotEncoder and RandomForestRegressor pipeline

The code to instantiate and fit the pipeline was:-

{% highlight python linenos %}
pipeline_randomforest_OHE = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True),
    RandomForestRegressor(n_estimators=100, random_state=42, verbose=1,n_jobs=-1)
)
pipeline_randomforest_OHE.fit(X_train, y_train)
{% endhighlight %}

Parameters to benchmark the model:-

| Parameter | Value |
| :------ |:--- |
| Time to fit the model | 222 sec |
| Training Score (R<sup>2</sup>)| 93.70 % |
| Validation Score (R<sup>2</sup>)| 47.76 % |
| Validation Score (R<sup>2</sup>)| 47.30 % |
| Baseline MAE | 3.379 mm
| Model MAE | 1.965 mm
| Improvement over Baseline MAE | 71.89 %

#### 3. OrdinalEncoder and XGBoost pipeline

Before the XGBoost pipeline can be instantiated and fit, train, validation and test dataset were updated as follows:-

Train - Data from 2001 - 2012
Validate - Data from 2013 - 2016
Test - Data from 2017 - 2020

This was done as XGBoost is able to fit the model much faster as compared with the RandomForestRegressor. 

The rest was similar to what was done in the past. The code to instantiated and fit the pipeline was:-

{% highlight python linenos %}
pipeline_xgboost = make_pipeline(
    ce.OrdinalEncoder(),
    XGBRegressor(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)
)
pipeline_xgboost.fit(X_train, y_train)
{% endhighlight %}

Parameters to benchmark the model:-

| Parameter | Value |
| :------ |:--- |
| Time to fit the model | 11.47 sec |
| Training Score (R<sup>2</sup>)| 59.00 % |
| Validation Score (R<sup>2</sup>)| 48.45 % |
| Test Score (R<sup>2</sup>)| 35.98 % |
| Baseline MAE | 3.379 mm
| Model MAE | 2.022 mm
| Improvement over Baseline MAE | 67.09 %

**Analysis**

Some correction in Model MAE was expected as RandomForestRegressor tries to fit a model with infinite depth. The model score for the RandomForestRegressor reflects this while the model wasn't doing very well with the validation score. Another thing to note in the XGBoost model is that the data for a longer duration was used compared to the earlier run models. This may be another source due to which our improvement over the baseline was reduced compared to the previous model.

### Features Importances from XGBRegressor

XGBRegressor was used to extract the top 15 features that contribute to the prediction of Precipitation.

![Top-15-Features](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Top-15-Features.png)
*Top 15 Features as determined after XGBRegressor Run*

### Permutation Importance from XGBRegressor

Permutation importance provides an insight in ranking the features of the data by permuting different values in any feature. Web bulb temperature still ranks the top in predicting the Precipitation, however, the ranking has changed for other factors, as shown in the output below.

![Permutation-Importance](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Permutation-Importance.png)
*Image Showing Permutation Importance by priority*

### Partial Dependence Plot (PDP)

A Partial Dependence Plot was built to see the effect of more than one feature on the predicted Precipitation. From the PDP, it can be inferred that the relationship between the features and the target is monotonic.

![PDP-Wet-Bulb_Temperature](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/PDP-Wet-Bulb_Temperature.png)
*Partial Dependence Plot showing the Variation of Precipitation with Wet Bulb Temperature*

![PDP-Wet-Bulb_Temperature-and-Radiative-Flux](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/PDP-Wet-Bulb_Temperature-and-Radiative-Flux.png)
*Partial Dependence Plot showing the Variation of Precipitation with Wet Bulb Temperature and Radiative Flux*

### Shap Values

Shap values are what the features contribute to the final predicted value.

![Shap Values](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Shap-Values.png)
*Image Showing how Precipitation Changes with Change in Features*

## Conclusion

Based on all the features used in this research, Temperature was the key feature that predicts Precipitation of any region. With temperatures rising globally due to global warming, the research in this project shows that the precipitation levels are also likely to increase. If it interests you in testing the XGBoost model to make predictions, then use the app in the link below.

**Add Link Here to Dash App**

## GitHub Repository

See the link below for the code used to get the DataFrame used in this research.
[https://github.com/singparvi/Colab_Notebooks/blob/master/Unit%202%20-%20Linear%20Models/Sprint_4_Build_Precipitation_Prediction/Get_Precipiration_Data_NASA_Submission.ipynb](https://github.com/singparvi/Colab_Notebooks/blob/master/Unit%202%20-%20Linear%20Models/Sprint_4_Build_Precipitation_Prediction/Get_Precipiration_Data_NASA_Submission.ipynb)

The code for the predictive modelling done in this project can be found in the link:-
[https://github.com/singparvi/Colab_Notebooks/blob/master/Unit%202%20-%20Linear%20Models/Sprint_4_Build_Precipitation_Prediction/Get_Precipiration_Data_NASA_Submission.ipynb](https://github.com/singparvi/Colab_Notebooks/blob/master/Unit%202%20-%20Linear%20Models/Sprint_4_Build_Precipitation_Prediction/Get_Precipiration_Data_NASA_Submission.ipynb)

## Sources

<sup>1</sup>[NASA POWER app](https://power.larc.nasa.gov/data-access-viewer/)

<sup>2</sup>[Latitude Longitude of Countries from albertyw](https://github.com/albertyw/avenews/blob/master/old/data/average-latitude-longitude-countries.csv)

<sup>3</sup>[Forest Cover Data from World Bank](https://data.worldbank.org/indicator/AG.LND.FRST.K2)