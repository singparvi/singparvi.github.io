---
layout: post
title: Regression Based Prediction Model for Precipitation using Global Weather Data from NASA from 2001 to 2020
subtitle: This project was to use Multiple Linear Regression (MLR) Machine Learning models on Global Weather Data from NASA to make precipitation prediction.
cover-img: /assets/img/Precipitation-Cover.jpg
thumbnail-img: /assets/img/Precipitation-Cover.jpg
share-img: /assets/img/Precipitation-Cover.jpg
tags: [python, MachineLearning, Precipitation, Prediction, model]
---
## Abstract

 Weather is changing globally and it has been never more important to make a better prediction on weather in the human history. There is a huge human and capital cost involved with a severe weather event. Current meteorological models can predict the weather in any area of the world with great accuracy. This research aims at predicting precipitation using historical information and leveraging Multiple Linear Regression (MLR) in python Machine Learning to make a prediction. The prediction does not have to be for the following week or month but it can be for a date many years in future. Multiple models were run using weather data from NASA and at best, an R^2 of 48.5% was obtained. The R^2 value may seem low but the Mean Absolute Error (MAE) has a 67% improvement from the baseline. The low R^2 is attributed to the non-availability of key weather information that could be added later to make a more refined prediction. Temperature was identified as the key feature in predicting precipitation. The predictions from the best model from this research may enable decision makers like governments or even insurance companies (with financial interests) make predictions to plan for policies or undertaken risks.

## Finding the Data and the JSON hurdle

Finding the data itself was a big hurdle in this project. I wanted to take up that provides me the learning opportunity and at the same time I could apply Machine Learning to the data set, the data must have enough observations to devote time on. At least a 100,000 so that I have enough observations to train, validate and then test. On the other hand the data must relate to something that can tie into a business case.

I finalized to take up the topic first to make prediction on rain and precipitation was the only thing that I found closest. NASA maintains an app called POWER Single Point Data Access <sup>1</sup> that provides data in JavaScript Object Notation (JSON) format through an Application Programming Interface (API) to users based on the geographical location of interest. NASA POWER app required lattitude and longitude to provide the weather information. This was achieved by using an existing CSV file from GitHub user albertyw <sup>2</sup>.

A python program was written in python notebook that takes in the lattitude longitude information from the country list and uses to fetch the following information from NASA application:-

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

The python code also merges the country code, lattitude and longitude data to make a single dataframe for use. The data included weather data for each day for 240 countries from 1980 to 2020. The resulting dataset had 3506400 rows × 22 columns.

###### insert data head here

The dataframe build here was used in the research further. The learning to now be able to use JSON data available publically, send JSON requests, receive and interpret  and convert them to pandas DataFrame was a small achievement in the process of the machine learning model.

---
## EDA and Machine Learning Model

NASA POWER data was cleaned and unnecessary or repetitive columns were dropped. Before we get into features and target selection another feature was included in the data that should have a factor affecting precipitation of any area of interest. Forest Area data from World Bank <sup>3</sup> was imported in the dataframe as a feature through pandas merge function.

Due to the resource limitations and for quick turnarounds in model training only last twenty years of data was considered.

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
17. Forest_Cover(sq KMs)

The definition of all the features mentioned above was provided in the text above. Precipitation was chosen as the target.

The target was skewed to the right due to the presence of some 300 observations. 

### Baseline

 Precipitation mean was chosen to set a baseline to compare the model performance. Precipitation mean was calculated for the entire data and was determined as **2.787 mm**. Mean Absolute Error (MAE) was calculated and was found to be **3.379 mm**. The baseline MAE is used to compare various models to see how are we better in making precipitation predictions.

### Models

Train, validate and test split was done. In the model pipelines that use more time and resources in fitting were allocated

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

Since the data was superclean with no missing values, compute or scaling were not used.

Parameters to benchmark the model:-

| Parameter | Value |
| :------ |:--- |
| Time to fit the model | 22 sec |
| Training Score (R<sup>2</sup>) | 93.70 % |
| Validation Score (R<sup>2</sup>) | 44.02 % |
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
| Baseline MAE | 3.379 mm
| Model MAE | 1.965 mm
| Improvement over Baseline MAE | 71.89 %

#### 3. OrdinalEncoder and XGBoost pipeline

Before the XGBoost pipeline can be instantiated and fit, train, validation and test dataset were updated as follows:-

Train - Data from 2001 - 2012
Validate - Data from 2013 - 2016
Test - Data from 2017 - 2020

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
| Baseline MAE | 3.379 mm
| Model MAE | 2.022 mm
| Improvement over Baseline MAE | 67.09 %

**Analysis**

Some correction in Model MAE was expectet as RandomForestRegressor tries to fit a model with infinite depth. This is reflected by the model score for the RandomForestRegressor while the model wasn't doing very well with the validation score. Another thing to note in the XGBoost model is that the data for a longer duration was used as compared to the earlier run models. Which may be another source due to which our improvement over the baseline was reduced.

### Features Importances from XGBRegressor

XGBRegressor was used to extract the top 15 features that contribute to the prediction of Precipitation.

![Top-15-Features](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Top-15-Features.png)
*Top 15 Features as determined after XGBRegressor Run*

### Permutation Importances from XGBRegressor

Permutation importances provides an insight in ranking the features if the data by permuting different values in any feature. Web bulb temperature still ranks the top in terms of predicting the precipitation however the ranking has changed for other factors as can be seen from the output below.

![Permutation-Importances](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Permutation-Importances.png)
*Image Showing Permutation Importances by priority*

### Partial Dependence Plot (PDP)

A Partial Dependence Plot was built to see the effect of more than one feature on the predicted Precipitation. From the PDP it can be inferred that the relationship between the features and the target is monotonic.

![PDP-Wet-Bulb_Temperature](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/PDP-Wet-Bulb_Temperature.png)
*Partial Dependence Plot showing the Variation of Precipitation with Wet Bulb Temperature*

![PDP-Wet-Bulb_Temperature-and-Radiative-Flux](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/PDP-Wet-Bulb_Temperature-and-Radiative-Flux.png)
*Partial Dependence Plot showing the Variation of Precipitation with Wet Bulb Temperature and Radiative Flux*
### Shap Values

Shap values are what the features contribute to the final predicted value.

![Shap Values](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Shap-Values.png)
*Image Showing how Precipitation Changes with Change in Features*
## Conclusion

Based on all the features used in this research, Temperature was found to be the key feature that helps in predicting Precipitation of any region with a positive correlation. With  temperatures rising global due to global warming it can be said from the research in this project that the precipitation levels are also likely to increase. If you are interesting in making any predictions using the XGBoost please refer to the app in the link below.

**Add Link Here to Dash App**

## GitHub Repository

See the link below for the code used to generate the population DataFrame.
[https://github.com/singparvi/US-Gun-Sales/tree/main/US_Population_Code](https://github.com/singparvi/US-Gun-Sales/tree/main/US_Population_Code)

The code for the graphics shown below could be in the link:-
[https://github.com/singparvi/US-Gun-Sales/blob/main/US_Gun_Sales_Code/US_Gun_Sales_Analysis.ipynb](https://github.com/singparvi/US-Gun-Sales/blob/main/US_Gun_Sales_Code/US_Gun_Sales_Analysis.ipynb)

## Sources

<sup>1</sup>[NASA POWER app](https://power.larc.nasa.gov/data-access-viewer/)

<sup>2</sup>[Latitude Longitude of Countries from albertyw]((https://github.com/albertyw/avenews/blob/master/old/data/average-latitude-longitude-countries.csv))

<sup>3</sup>[Forest Cover Data from World Bank](https://data.worldbank.org/indicator/AG.LND.FRST.K2)