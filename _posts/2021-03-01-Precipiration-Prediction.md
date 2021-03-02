---
layout: post
title: Precipitation Prediction using Global Weather Data from NASA from 2001 to 2020
subtitle: This project was to use Global Weather Data from NASA and make precipitation prediction using Machine Learning models.
cover-img: /assets/img/Precipitation-Cover.jpg
thumbnail-img: /assets/img/UPrecipitation-Cover.jpg
share-img: /assets/img/Precipitation-Cover.jpg
tags: [python, MachineLearning, Precipitation, Prediction, model]
---
## Topic Selection and the Research Questions

Before the 2020 US election, a Washington Post News Article <sup>1</sup> caught my eye. I started to investigate whether the 2020 election has anything to do with gun sales in the US. Was this year unique in any way due to COVID, or perhaps the sitting president? This got me started in collecting data on US gun sales and analyzing it. 

## Source Dataset

The data for this research was obtained from the National Instant Criminal Background Check System (NICS)<sup>2</sup> database maintained by the FBI. The NICS system was launched in 1998 and allowed a Federal Firearms Licensee (FFL), an authorized seller in the US, to run a background check on the buyer. Since its inception, NICS has provided approximately 300 million checks and led to 1.5 million denials. The denials may represent only 0.5%, but the data reveals which states are buying most guns in the US.  NICS reports the data for each state monthly in a pdf format. Buzzfeednews.com maintains a CSV format of the data from the FBI, which was of great help in the data analysis.<sup>3</sup>

In addition to the NICS database, the US population data of each state was also used. Information from census.gov <sup>4</sup> and jakevdp from GitHub data <sup>5</sup> was used to determine state populations from 2001-2019. The state populations for 2020 was extrapolated from 2018 and 2019 numbers.

---
## Visualization and Data Interpretation

For data analysis, only the data from 2001-2020 was considered. The line plot shown below displays the gun sales month by month across the US from 2001-2020. Key events explain major spikes in gun sales. There is an increase in gun sales around each election, be it 2008, 2012, 2016 or 2020.
The US buys guns mainly in December, consistently year after year.

![Annual Gun Sales Events](https://github.com/singparvi/singparvi.github.io/raw/8ecf4bc80cf1feceac6fbf9a9699a69799a41335/assets/img/US_Annual_Gun_Sales_Events.jpeg)

---
## Non-Normalized Data
The following section's data is not normalized, and the intention is to show that some states might appear to be buying a lot of guns, but the states of interest change when accounting for the population.

### Total Annual Gun Sales in US States by Year

The map below shows how total gun sales have changed in the US by state each year. The color intensity shows the state that has the maximum gun sales in any time frame. 

{% include Choropleth_Annual_Gun_Sales_Population.html %}
### US Gun Sales from 2001 to 2020

2016 and 2020, both election years, had record sales and there is a significant spike in gun sales in both years. 2020 is on track to be a record year in gun sales with more than 30 million guns sold as of November 2020.

{% include Barplot_YoY_Gun_Sales_Population.html %}

### Total Gun Sales per US state from 2001 to 2020

To more concisely see the scale of change the bar plot below shows how gun sale patterns shift over time. In 2001, California accounted for the most gun sales while for 2020, Illinois has taken the spot.

{% include Barplot_Annual_Gun_Sales_Population.html %}

### Gun Sales from 2001 to 2020 of top 10 US States

The bar chart below shows how gun sales intensity shifted from California to Illinois in the past two decades. Press the play button to see how states change rank over time. Surprisingly California has seen a steady decline in gun sales between 2017-2019 with a little increase in 2020.

{% include BarChartRace_Annual_Gun_Sales_Population.html %}

---

## Normalized data

As can be seen above, in 2020 Illinois had the largest total gun sales. However, the statistics change when the population is considered.

### Total Annual Gun Sales per 100k Population of US States by Year

The map below shows the number of guns sold per 100k population of any state in any given year. In 2001, Colorado, Montana and West Virginia accounted for most guns sold in the US with 82, 78 and 76 guns per 100k individuals respectively. Fast forward twenty years, Kentucky, Illinois and Utah have taken these spots. Kentucky has approximately 651 guns for every 100,000 individuals while Illinois at 520 guns for every 100,000 individuals. Overall there has been a dramatic shift in gun sales over the years. 

{% include Choropleth_Annual_Gun_Sales_100k.html %}

### US Gun Sales with Population Consideration

To more concisely see the scale of change, the bar plot shows the information in a bar plot below shows the change in scale in the past twenty years. 

{% include Barplot_Annual_Gun_Sales_100k.html %}

### Bar chart race for Annual Gun Sales per 100k population from 2001 to 2020  of top 10 US States 

Kentucky has held the first spot since 2006 for the highest number of guns sold per 100k individuals. That year the number of guns purchased per 100k individuals was almost four times greater than 2005. The number of guns purchased in Kentucky in 2007 was almost double the number of guns bought in 2006!.

{% include BarChartRace_Annual_Gun_Sales_100k.html %}

---
## Correlation and Linear Regression Fitting

The states which bought the most guns per 100k individuals in 2020 were Kentucky, Illinois and Utah.  

### Kentucky State Correlation and Linear Regression Fitting

The correlation between Kentucky state's guns sold per 100k and its population is approximately 91%. With linear regression, it can be interpreted that the gun sales per 100k in Kentucky can be explained approximately 83% by time only.

![Kentucky OLS Results](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Kentucky-OLS-Results.png)
### Illinois State Correlation and Linear Regression Fitting

The correlation between Illinois state's guns sold per 100k and its population is approximately 75%. With linear regression, it can be interpreted that the gun sales per 100k in Illinois can be explained approximately 54% by time only.

![Illinois OLS Results](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Illinois-OLS-Results.png)

### Utah State Correlation and Linear Regression Fitting

The correlation between Utah state's guns sold per 100k and its population is approximately 59%. With linear regression, it can be interpreted that the gun sales per 100k in Utah can be explained approximately 31% by time only.

![Utah OLS Results](https://raw.githubusercontent.com/singparvi/singparvi.github.io/master/assets/img/Utah-OLS-Results.png)

## Conclusion
 
The NICS data provide an excellent insight into the gun sales trend in the US. Election years are related to an increase in US gun sales. The sitting president might be why this factor was amplified for 2020. For future election years, it is a higher probability that US gun sales will see a distinct increase in sales compared to the preceding year. At the beginning of the data analysis, I hypothesized that the coronavirus situation impacted gun sales with unemployment. The data helped me reject the hypothesis.
## GitHub Repository

See the link below for the code used to generate the population DataFrame.
[https://github.com/singparvi/US-Gun-Sales/tree/main/US_Population_Code](https://github.com/singparvi/US-Gun-Sales/tree/main/US_Population_Code)

The code for the graphics shown below could be in the link:-
[https://github.com/singparvi/US-Gun-Sales/blob/main/US_Gun_Sales_Code/US_Gun_Sales_Analysis.ipynb](https://github.com/singparvi/US-Gun-Sales/blob/main/US_Gun_Sales_Code/US_Gun_Sales_Analysis.ipynb)

## Sources

<sup>1</sup>[Washington Post News Article](https://www.washingtonpost.com/business/2020/10/29/walmart-guns-civil-unres/)

<sup>2</sup>[FBI NICS](https://www.fbi.gov/services/cjis/nics) 

<sup>3</sup>[BuzzFeedNews](https://github.com/BuzzFeedNews/nics-firearm-background-checks)

<sup>4</sup>[Census U.S.](http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv)

<sup>5</sup>[GitHub Link](https://github.com/jakevdp/data-USstates)

