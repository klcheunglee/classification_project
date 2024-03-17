#!/usr/bin/env python
# coding: utf-8

# # INN Hotels Project
# 
# ## Context
# 
# A significant number of hotel bookings are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.
# 
# The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.
# 
# The cancellation of bookings impact a hotel on various fronts:
# * Loss of resources (revenue) when the hotel cannot resell the room.
# * Additional costs of distribution channels by increasing commissions or paying for publicity to help sell these rooms.
# * Lowering prices last minute, so the hotel can resell a room, resulting in reducing the profit margin.
# * Human resources to make arrangements for the guests.
# 
# ## Objective
# The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. INN Hotels Group has a chain of hotels in Portugal, they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.
# 
# ## Data Description
# The data contains the different attributes of customers' booking details. The detailed data dictionary is given below.
# 
# 
# **Data Dictionary**
# 
# * Booking_ID: unique identifier of each booking
# * no_of_adults: Number of adults
# * no_of_children: Number of Children
# * no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# * no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# * type_of_meal_plan: Type of meal plan booked by the customer:
#     * Not Selected – No meal plan selected
#     * Meal Plan 1 – Breakfast
#     * Meal Plan 2 – Half board (breakfast and one other meal)
#     * Meal Plan 3 – Full board (breakfast, lunch, and dinner)
# * required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# * room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# * lead_time: Number of days between the date of booking and the arrival date
# * arrival_year: Year of arrival date
# * arrival_month: Month of arrival date
# * arrival_date: Date of the month
# * market_segment_type: Market segment designation.
# * repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# * no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# * no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# * avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# * no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# * booking_status: Flag indicating if the booking was canceled or not.

# ## Importing necessary libraries and data

# In[69]:


get_ipython().system('pip install -U scikit-learn')


# In[70]:


get_ipython().system('pip install scikit-plot')


# In[71]:


import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Library to split data
from sklearn.model_selection import train_test_split

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get different metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)
import scikitplot as skplt


# ## Import Dataset

# In[72]:


hotel = pd.read_csv('/content/INNHotelsGroup.csv')


# In[73]:


# copying data to another variable to avoid any changes to original data
data = hotel.copy()


# ## Data Overview
# 
# - Observations
# - Sanity checks

# ### View the first and last 5 rows of the dataset

# In[74]:


data.head()


# In[75]:


data.tail()


# ### Understand the shape of the dataset

# In[76]:


data.shape


# - There are 19 columns and 36275 rows in the dataset.

# ### Check the data types of the columns for the dataset

# In[77]:


data.info()


# - **`Booking_ID`**, **`type_of_meal_plan`**,**`room_type_reserved`**, **`market_segment_type`**, and **`booking_status`** are of object type while rest columns are numeric (int64 / float64).
# - All columns are non-null.

# In[78]:


# checking for duplicate values
data.duplicated().sum()


# **Let's drop the Booking_ID column first before we proceed forward**.

# In[79]:


data = data.drop(["Booking_ID"], axis=1)


# In[80]:


data.head()


# The **`Booking_ID`** column is now dropped.

# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - A thorough analysis of the data, in addition to the questions mentioned below, should be done.

# **Leading Questions**:
# 1. What are the busiest months in the hotel?
# 2. Which market segment do most of the guests come from?
# 3. Hotel rates are dynamic and change according to demand and customer demographics. What are the differences in room prices in different market segments?
# 4. What percentage of bookings are canceled?
# 5. Repeating guests are the guests who stay in the hotel often and are important to brand equity. What percentage of repeating guests cancel?
# 6. Many guests have special requirements when booking a hotel room. Do these requirements affect booking cancellation?

# **Let's check the statistical summary of the data.**

# In[81]:


data.describe().T


# - The **`no_of_adults`** column ranges from 0 to 4.
# - The maximum value in the **`no_of_children`** column is 10, which seems a bit unusual and may require further investigation.
# - The values in the **`no_of_weekend_nights`** and **`no_of_week_nights`** columns appear to be reasonable. However, a stay of 7 weekends seems exceptionally long.
# - Interestingly, at least 75% of the customers do not require car parking space, which could be attributed to the hotel's location.
# - Regarding the **`lead_time`** column, there is a significant disparity between the 75th percentile and the maximum value, indicating the possible presence of outliers in this column.
# - It is worth noting that at least 75% of the customers are not repeated customers.
# - The average price per room is 103 euros, but there is a substantial difference between the 75th percentile and the maximum value, suggesting the potential presence of outliers in this column.

# ### Univariate Analysis

# In[82]:


def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="pink"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, color="violet"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, color="violet"
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# ### Observations on **`lead_time`**

# In[83]:


histogram_boxplot(data, "lead_time")


# Observations:
# - There are numerous outliers on the right side of the **`lead_time`** distribution.
# - The distribution of **`lead_time`** is skewed to the right, indicating a longer tail on that side.
# - Interestingly, there are guests who have made bookings more than 400 days in advance, which is equivalent to more than a year ago.
# - It is notable that a significant number of guests have booked on the same day as their arrival.

# ### Observations on **`avg_price_per_room`**

# In[84]:


histogram_boxplot(data, "avg_price_per_room")


# Observations:
# - There are numerous outliers on the both sides of the **`avg_price_per_room`** distribution.
# - The distribution of **`avg_price_per_room`** is skewed to the right, indicating a longer tail on that side.
# - The average price per room is approximately €103.4.
# - An intriguing observation is that there are some rooms with a price of €0.
# - We have identified a room with a cost of €540, which stands out as a significant difference. Rather than dropping it, we will limit this value to the upper whisker (Q3 + 1.5 * IQR).

# Firstly, let's check the rooms with a price of €0.

# In[85]:


data[data["avg_price_per_room"] == 0]


# - It is worth noting that there are 545 records where the room cost is €0.
# - Interestingly, many of them's market segment are recorded as 'complimentary service'.

# In[86]:


data.loc[data["avg_price_per_room"] == 0, "market_segment_type"].value_counts()


# - The presence of 545 rooms with a free room cost can be attributed to different reasons. Out of these, 354 rooms are complementary services provided by the hotel, where the customers are offered a free stay. As for the remaining rooms, it is possible that they are part of an online promotion or offer.

# - Next, we will address the outlier, specifically the room with a cost of €540.

# In[87]:


# Calculating the 25th quantile
Q1 = data["avg_price_per_room"].quantile(0.25)

# Calculating the 75th quantile
Q3 = data["avg_price_per_room"].quantile(0.75)

# Calculating IQR
IQR = Q3 - Q1

# Calculating value of upper whisker
Upper_Whisker = Q3 + 1.5 * IQR
Upper_Whisker


# In[88]:


# assigning the outliers the value of upper whisker
data.loc[data["avg_price_per_room"] >= 500, "avg_price_per_room"] = Upper_Whisker


# ### Observations on **`no_of_previous_cancellations`**

# In[89]:


histogram_boxplot(data, "no_of_previous_cancellations")


# Observations:
# - Most of the customers do not have previous cancellation record.
# - There are some customers who had 13 previous cancellation record, which is a bit unusual.

# In[90]:


data.loc[data["no_of_previous_cancellations"] == 13]


# - It is highly likely that these 4 rows are duplicates since all values in every column are identical.

# ### Observations on **`no_of_previous_bookings_not_canceled`**

# In[91]:


histogram_boxplot(data, "no_of_previous_bookings_not_canceled")


# Observations:
# 
# - Very few customers have more than 1 booking not canceled previously.
# - Some customers have not canceled their bookings for 58 times.

# In[92]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# ### Observations on **`no_of_adults`**

# In[93]:


labeled_barplot(data, "no_of_adults", perc=True)


# Observations:
# - 72% of the bookings consist of 2 adults.

# ### Observations on **`no_of_children`**

# In[94]:


labeled_barplot(data, "no_of_children", perc = True)


# Observations:
# - 92.6% of the bookings consist of 0 child.
# - There are some bookings that include 9 or 10 children, which is quite unusual. For the purpose of analysis, we will replace these data points with the maximum value.

# In[95]:


# replacing 9, and 10 children with 3
data["no_of_children"] = data["no_of_children"].replace([9, 10], 3)


# In[96]:


labeled_barplot(data, "no_of_children", perc = True)


# - We successfully replaced it.

# ### Observations on **`no_of_week_nights`**

# In[97]:


labeled_barplot(data, "no_of_week_nights", perc = True)


# Observations:
# - The analysis shows that 31.5% of bookings are made for a 2-weekday night stay, while 26.2% of bookings are made for a 1-weekday night stay.
# - There are very few bookings that involve a stay of more than 10-weekday nights.

# ### Observations on **`no_of_weekend_nights`**

# In[98]:


labeled_barplot(data, "no_of_weekend_nights", perc = True)


# Observations:
# 
# - The analysis shows that 46.5% of bookings are made for a 0-weekend night stay, while 27.6% of bookings are made for a 1-weekend night stay.
# - The percentage of customers planning to spend either 1 or 2 weekends in the hotel is nearly equal.
# - There are very few bookings that involve a stay of 7-weekend nights.

# ### Observations on **`required_car_parking_space`**

# In[99]:


labeled_barplot(data, "required_car_parking_space", perc = True)


# Observations:
# - 96.9% of guests do not require car parking space.

# ### Observations on **`type_of_meal_plan`**

# In[100]:


labeled_barplot(data, "type_of_meal_plan", perc = True)


# Observations:
# - 76.7% of guests selected meal plan 1: breakfast only for their stay.
# - 14.1% of guest did not select any meal plans for their stay.

# ### Observations on **`room_type_reserved`**

# In[101]:


labeled_barplot(data, "room_type_reserved", perc = True)


# Observations:
# - 77.5% of bookings are made for room type 1, followed by 16.7% of bookings are made for room type 4.

# ### Observations on **`arrival_month`**

# In[102]:


labeled_barplot(data, "arrival_month", perc= True)


# Observations:
# 
# - 14.7% of bookings are made on October, followed by 12.7% of bookings are made on September.

# ### Observations on **`market_segment_type`**

# In[103]:


labeled_barplot(data, "market_segment_type", perc=True)


# Observations:
# 
# - 64.0% of bookings are made online, followed by 29.0% of bookings are made offline.

# ### Observations on **`no_of_special_requests`**

# In[104]:


labeled_barplot(data, "no_of_special_requests", perc = True)


# Observations:
# 
# - 54.5% of bookings did not have any special requests, followed by 31.4% of bookings are made for 1 special request.

# ### Observations on **`booking_status`**

# In[105]:


labeled_barplot(data, "booking_status", perc = True)


# Observations:
# 
# - 67.2% of bookings were not canceled, followed by 32.8% of bookings were canceled.

# **Let's encode Canceled bookings to 1 and Not_Canceled as 0 for further analysis**

# In[106]:


data["booking_status"] = data["booking_status"].apply(
    lambda x: 1 if x == "Canceled" else 0
)


# In[107]:


labeled_barplot(data, "booking_status", perc = True)


# - We successfully encoded the booking status.

# ### Bivariate Analysis

# In[108]:


cols_list = data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(
    data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral"
)
plt.show()


# Observations:
# - There is a positive correlation (0.54) between **`repeated_guest`** and **`no_of_previous_bookings_not_canceled`**. It suggests that the behavior of not canceling previous bookings might contribute to building a base of loyal customers who choose to stay at the hotel repeatedly. It could imply that these customers have had positive experiences in the past and continue to choose the hotel for their future stays.
# - There is a positive correlation (0.47) between **`no_of_previous_cancellations`** and **`no_of_previous_bookings_not_canceled`** by a customer, indicating that there is a tendency for customers who have a higher number of previous cancellations to also have a higher number of previous bookings that were not canceled. It could imply that some customers might have experienced changes in their travel plans or other reasons for cancellations in the past but have also demonstrated their commitment to the hotel by not canceling other bookings.
# - There is a positive correlation (0.44) between the **`booking_status`** and **`lead_time`**, indicating that a longer lead time is associated with a higher likelihood of cancellation. Further analysis will be conducted to delve deeper into this relationship.
# - The analysis reveals a positive correlation between the number of customers (**`no_of_adults`** : 0.30 and **`no_of_children`** : 0.35) and the **`avg_price_per_room`**. Because of the increase in the number of customers would require more rooms, consequently raising the overall cost.
# - A positive correlation (0.15) exists between the **`lead_time`** and **`no_of_week_nights`** a customer plans to stay in the hotel. This implies that customers who plan longer stays tend to book their accommodations well in advance.
# - A negative correlation is observed between the **`avg_price_per_room`** and **`repeated_guest`** (-0.18). This suggests that the hotel might offer loyalty benefits to incentivize repeat bookings.
# - A negative correlation (-0.25) is observed between the **`no_of_special_requests`** from the customer and the **`booking_status`**. This suggests that if a customer makes special requests, the chances of cancellation may decrease. Additional analysis will be performed to explore this correlation in more detail.

# **Creating functions that will help us with further analysis.**

# In[109]:


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

    # Calculate the statistical summary
    summary = data.groupby(target)[predictor].describe()
    print(summary)


# In[110]:


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# **Hotel rates are dynamic and change according to demand and customer demographics. Let's see how prices vary across different market segments**

# In[111]:


plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data, x="market_segment_type", y="avg_price_per_room", palette="gist_rainbow"
)
plt.show()


# Observations:
# - Among different market segment types (offline, corporate, aviation & complementary), online bookings show a noteworthy trend, with 75% of them having an average room price higher than the other segments. This suggests that online bookings tend to command relatively higher prices compared to bookings made through offline channels, corporate clients, aviation-related bookings, and complementary services.
# - Compared to other booking types, most complementary services bookings have a lower average room price. This implies that customers availing complementary services generally pay less for their accommodations compared to bookings made through other segments..
# 

# **Let's see how booking status varies across different market segments. Also, how average price per room impacts booking status**

# In[112]:


stacked_barplot(data, "market_segment_type", "booking_status")


# Observations:
# 
# - The "Online" segment has the highest number of bookings, with 14,739 bookings that were not canceled and 8,475 bookings that were canceled. This indicates that a significant proportion of bookings made through the online channel resulted in cancellations.
# - The "Offline" segment also has a considerable number of bookings, with 7,375 bookings that were not canceled and 3,153 bookings that were canceled. This suggests that both online and offline channels are popular choices for making bookings, but the cancellation rate is relatively lower in the offline segment compared to the online segment.
# - The "Corporate" segment shows a lower number of bookings in comparison, with 1,797 bookings that were not canceled and 220 bookings that were canceled. This indicates that the corporate segment contributes a smaller portion of the overall bookings, but the cancellation rate is relatively low within this segment.
# - The "Aviation" segment has the lowest number of bookings, with 88 bookings that were not canceled and 37 bookings that were canceled. This suggests that the aviation segment represents a smaller portion of the bookings, and the cancellation rate is relatively higher compared to other segments.
# - The "Complementary" segment has 391 bookings, and interestingly, none of them were canceled. This indicates that the hotel provided these bookings as complementary services to customers, and they were honored without any cancellations.

# **Many guests have special requirements when booking a hotel room. Let's see how it impacts cancellations**

# In[113]:


stacked_barplot(data, "no_of_special_requests", "booking_status")


# Observations:
# - The cancellation rate is observed to be higher among guests who did not make any special requests.
# - On the other hand, guests who made more than 2 special requests have a tendency to not cancel their room reservations.
# - It indicate a potential relationship between the number of special requests made by guests and their likelihood of canceling their bookings. Guests who did not make any special requests might be more inclined to cancel, while guests with multiple special requests may have a higher commitment to their reservations, leading to a lower cancellation rate.

# **Let's see if the special requests made by the customers impacts the prices of a room**

# In[114]:


plt.figure(figsize=(10, 5))
sns.boxplot(
    data=data,
    x="no_of_special_requests",
    y="avg_price_per_room",
    showfliers=False,  # excluding the outliers
)
plt.show()


# Observations:
# 
# - As the number of special requests increases from 0 to 2, there is a gradual increase in the mean **`avg_price_per_room`**. This suggests that customers who make more special requests tend to have higher room prices on average.
# - However, for customers with 3 or 4 special requests, the mean **`avg_price_per_room`** remains relatively stable, indicating that there might not be a significant impact on prices beyond a certain threshold of special requests.
# - The interquartile range (IQR) for each category gradually widens as the number of special requests increases. This indicates increasing variability in room prices with more special requests.
# - The minimum and maximum values also show a gradual increase as the **`no_of_special_requests`** increases, indicating the presence of higher-priced rooms in categories with more special requests.
# - It is important to note that other factors may also influence room prices, and further analysis is needed to determine the exact nature and extent of the impact.

# **We saw earlier that there is a positive correlation between booking status and average price per room. Let's analyze it**

# In[115]:


distribution_plot_wrt_target(data, "avg_price_per_room", "booking_status")


# Observations:
# 
# - For bookings with a status of 0 (not canceled), the average price per room is approximately €99.93. The prices range from €0 to €375.50, with a standard deviation of approximately €35.87. The majority of prices fall within the range of €77.86 to €119.10, as indicated by the 25th and 75th percentiles.
# - On the other hand, for bookings with a status of 1 (canceled), the average price per room is higher at around €110.56. The prices range from €0 to €365.00, with a standard deviation of approximately €32.03. The majority of prices fall within the range of €89.27 to €126.36, as indicated by the 25th and 75th percentiles.
# - Based on this information, we can infer that there is a positive correlation between booking status and the average price per room. On average, canceled bookings tend to have slightly higher room prices compared to bookings that are not canceled.

# **There is a positive correlation between booking status and lead time also. Let's analyze it further**

# In[116]:


distribution_plot_wrt_target(data, "lead_time", "booking_status")


# Observations:
# 
# - For bookings with a status of 0 (not canceled), the average lead time is approximately 58.93 days. The lead time ranges from 0 to 386 days, with a standard deviation of approximately 64.03 days. The majority of lead times fall within the range of 10 to 86 days, as indicated by the 25th and 75th percentiles.
# 
# - On the other hand, for bookings with a status of 1 (canceled), the average lead time is higher at around 139.22 days. The lead time ranges from 0 to 443 days, with a standard deviation of approximately 98.95 days. The majority of lead times fall within the range of 55 to 205 days, as indicated by the 25th and 75th percentiles.
# 
# - Based on this information, we can infer that there is a positive correlation between booking status and lead time. Canceled bookings tend to have longer lead times compared to bookings that are not canceled. This suggests that the longer the lead time, the higher the chances of a booking being canceled.

# **Generally people travel with their spouse and children for vacations or other activities. Let's create a new dataframe of the customers who traveled with their families and analyze the impact on booking status.**

# In[117]:


family_data = data[(data["no_of_children"] >= 0) & (data["no_of_adults"] > 1)]
family_data.shape


# In[118]:


family_data["no_of_family_members"] = (
    family_data["no_of_adults"] + family_data["no_of_children"]
)


# In[119]:


stacked_barplot(family_data, "no_of_family_members", "booking_status")


# Observations:
# 
# - In total, there are 28,441 customers who traveled with their families. Out of these, 18,456 bookings were not canceled (booking status 0), and 9,985 bookings were canceled (booking status 1).
# 
# - Among the customers who traveled with their families, the majority (23,719 bookings) had a family size of 2. Out of these, 15,506 bookings were not canceled, and 8,213 bookings were canceled.
# 
# - There were 3,793 bookings made by customers with a family size of 3. Out of these, 2,425 bookings were not canceled, and 1,368 bookings were canceled.
# 
# - For customers with a family size of 4, there were 912 bookings. Out of these, 514 bookings were not canceled, and 398 bookings were canceled.
# 
# - Only a small number of customers (17 bookings) had a family size of 5. Among these, 11 bookings were not canceled, and 6 bookings were canceled.
# 
# - From this data, we can observe that the majority of customers who traveled with their families had a family size of 2. Additionally, the number of cancellations is higher for larger family sizes, with a higher percentage of cancellations observed for families of size 4 and 5.

# **Let's do a similar analysis for the customer who stay for at least a day at the hotel.**

# In[120]:


stay_data = data[(data["no_of_week_nights"] > 0) & (data["no_of_weekend_nights"] > 0)]
stay_data.shape


# In[121]:


stay_data["total_days"] = (
    stay_data["no_of_week_nights"] + stay_data["no_of_weekend_nights"]
)


# In[122]:


stacked_barplot(stay_data, "total_days", "booking_status")


# Observations:
# 
# - In total, there are 17,094 customers who stayed at least one day at the hotel. Out of these, 10,979 bookings were not canceled (booking status 0), and 6,115 bookings were canceled (booking status 1).
# 
# - The majority of customers (5,872 bookings) stayed for 3 days at the hotel. Out of these, 3,689 bookings were not canceled, and 2,183 bookings were canceled.
# 
# - There were 4,364 bookings made by customers who stayed for 4 days. Out of these, 2,977 bookings were not canceled, and 1,387 bookings were canceled.
# 
# - For customers who stayed for 5 days, there were 2,331 bookings. Out of these, 1,593 bookings were not canceled, and 738 bookings were canceled.
# 
# - Similarly, for customers who stayed for 2, 6, 7, and 8 days, there were 1,940, 1,031, 973, and 179 bookings respectively. The number of bookings not canceled and canceled varied for each duration.
# 
# - There were also bookings for longer durations, such as 10, 9, 14, 15, 13, 12, 11, and so on. The number of bookings and their cancellation status varied for these durations as well.
# 
# - From this analysis, we can observe that the majority of customers stayed for 3 or 4 days at the hotel. The number of cancellations varies across different durations, with a higher percentage of cancellations observed for longer stays.

# **Repeating guests are the guests who stay in the hotel often and are important to brand equity. Let's see what percentage of repeating guests cancel?**

# In[123]:


stacked_barplot(data, "repeated_guest", "booking_status")


# Observations:
# 
# - In total, there are 36,275 bookings made by both repeating and non-repeating guests. Out of these, 24,390 bookings were not canceled (booking status 0), and 11,885 bookings were canceled (booking status 1).
# 
# - Among all the bookings, 930 were made by repeating guests, out of which only 16 bookings were canceled.
# 
# - For non-repeating guests, there were 35,345 bookings, out of which 11,869 bookings were canceled.
# 
# To calculate the cancellation percentage for repeating guests, we can divide the number of canceled bookings by the total number of bookings made by repeating guests and multiply by 100:
# 
# - Cancellation percentage for repeating guests = (16 / 930) * 100 ≈ 1.72%
# 
# - Therefore, approximately 1.72% of repeating guests canceled their bookings.
# 
# - Repeating guests, who are considered important for brand equity, have a relatively low cancellation rate compared to non-repeating guests. This suggests that repeating guests may have a higher level of commitment and loyalty to the hotel, leading to a lower likelihood of cancellations.

# **Let's find out what are the busiest months in the hotel.**

# In[124]:


# grouping the data on arrival months and extracting the count of bookings
monthly_data = data.groupby(["arrival_month"])["booking_status"].count()

# creating a dataframe with months and count of customers in each month
monthly_data = pd.DataFrame(
    {"Month": list(monthly_data.index), "Guests": list(monthly_data.values)}
)

# plotting the trend over different months
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_data, x="Month", y="Guests")
plt.show()


# Observations:
# - From April to July, the number of bookings at the hotel remains relatively consistent, with an average of around 3,000 to 3,500 guests during this period.
# - In October, the hotel experiences a significant increase in bookings, with more than 5,000 bookings recorded.
# - January sees the lowest number of bookings, with only around 1,000 bookings made.
# - This trend indicates that October is the busiest month for the hotel in terms of bookings, while January has the fewest bookings compared to other months.

# **Let's check the percentage of bookings canceled in each month.**

# In[125]:


stacked_barplot(data, "arrival_month", "booking_status")


# Observations:
# 
# Lowest Cancellation Percentage:
# - January: 2.37% of bookings were canceled.
#   - out of a total of 1,014 bookings, 990 (approximately 97.6%) were not canceled, while 24 (approximately 2.4%) were canceled.
# - December: 13.29% of bookings were canceled
#   - out of a total of 3,021 bookings, 2,619 (approximately 86.7%) were not canceled, while 402 (approximately 13.3%) were canceled.
# - February: 25.29% of bookings were canceled
#   - out of a total of 1,704 bookings, 1,274 (approximately 74.8%) were not canceled, while 430 (approximately 25.2%) were canceled
# 
# Highest Cancellation Percentage:
# - July: 44.9% of bookings were canceled
#   - out of a total of 2,920 bookings, 1,606 (approximately 55.1%) were not canceled, while 1,314 (approximately 44.9%) were canceled.
# -June: 40.3% of bookings were canceled
#   - out of a total of 3,203 bookings, 1,912 (approximately 59.7%) were not canceled, while 1,291 (approximately 40.3%) were canceled.
# - August: 39% of bookings were canceled
#   - out of a total of 3,813 bookings, 2,325 (approximately 61%) were not canceled, while 1,488 (approximately 39%) were canceled.
# 
# Even though the highest number of bookings were made in September and October - more than 30% of these bookings got canceled.
# - In October, out of a total of 5,317 bookings, 3,437 (approximately 64.6%) were not canceled, while 1,880 (approximately 35.4%) were canceled.
# - In September, out of a total of 4,611 bookings, 3,073 (approximately 66.7%) were not canceled, while 1,538 (approximately 33.3%) were canceled.
# 

# **As hotel room prices are dynamic, Let's see how the prices vary across different months**

# In[126]:


plt.figure(figsize=(10, 5))
sns.lineplot(y=data["avg_price_per_room"], x=data["arrival_month"], ci=None)
plt.show()


# The **`avg_price_per_room`** is highest during the months of May to September, with approximately €115 per room.

# ### Outlier Check
# 
# - Let's check for outliers in the data.

# In[127]:


# outlier detection using boxplot
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
# dropping booking_status
numeric_columns.remove("booking_status")

plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()


# - We have observed that there are several outliers present in the data. - However, we have decided not to remove or treat them as they are proper values.

# ####**EDA after data manipulation**

# In[209]:


data.describe().T


# In[210]:


# check data has no missing values as required
data.isnull().sum()


# In[211]:


# take a look at how data looks like now
data.head()


# In[212]:


data.dtypes


# ## Data Preperation for modeling
# 

# - To predict which bookings will be canceled, we need to encode the categorical features in the data.
# - Before building a predictive model, we will split the data into train and test sets.
# - This will allow us to evaluate the performance of the model on the train data and assess its generalization on the test data.

# In[128]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True)

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[129]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# - We had seen that around 67% of observations belongs to class 0 (Booking Not canceled) and arund 32% observations belongs to class 1 (Booking Canceled), and this is preserved in the train and test sets

# ## Model Building

# ### Model evaluation criterion
# 
# ### Model can make wrong predictions as:
# 
# 1. Predicting a customer will not cancel their booking but in reality, the customer will cancel their booking.
# 2. Predicting a customer will cancel their booking but in reality, the customer will not cancel their booking.
# 
# ### Which case is more important?
# * Both the cases are important as:
# 
# * If we predict that a booking will not be canceled and the booking gets canceled then the hotel will lose resources and will have to bear additional costs of distribution channels.
# 
# * If we predict that a booking will get canceled and the booking doesn't get canceled the hotel might not be able to provide satisfactory services to the customer by assuming that this booking will be canceled. This might damage the brand equity.
# 
# 
# 
# ### How to reduce the losses?
# 
# * Hotel would want `F1 Score` to be maximized, greater the F1  score higher are the chances of minimizing False Negatives and False Positives.

# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_statsmodels function will be used to check the model performance of models.
# * The confusion_matrix_statsmodels function will be used to plot the confusion matrix.

# In[130]:


# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification_statsmodels(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred_temp = model.predict(predictors) > threshold
    # rounding off the above values to get classes
    pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[131]:


# defining a function to plot the confusion_matrix of a classification model


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Logistic Regression (with statsmodels library)

# #### Data Preparation for modeling (Logistic Regression)

# - We want to predict which bookings will be canceled.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# #### Building Logistic Regression Model

# In[132]:


# fitting logistic regression model
logit = sm.Logit(y_train, X_train.astype(float))
lg = LogisticRegression(solver="newton-cg", random_state=1)

print(lg.fit(X_train, y_train))


# ### Checking performance on training set

# In[133]:


print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# ### Checking performance on test set

# In[136]:


print("Test performance:")
model_performance_classification_statsmodels(lg, X_test, y_test)


# Observations:
# - The F1 score on both the training set (approximately 0.68) and test set (approximately 0.68) is very close, suggesting that our model is not overfitting and is performing well.
# - The model demonstrates high precision (around 0.74 for the training set and 0.73 for the test set) but relatively low recall (approximately 0.63 for both the training and test sets).
# - For a good F1 score, it is desirable to have comparable precision and recall values.

# In[139]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

# adding constant
X = sm.add_constant(X)

X = pd.get_dummies(X, drop_first=True)

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[140]:


# fitting logistic regression model
logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(disp=False)

print(lg.summary())


# In[141]:


print("Training performance:")
model_performance_classification_statsmodels(lg, X_train, y_train)


# Observations:
# - Negative coefficients indicate that an increase in the corresponding attribute value decreases the probability of customers canceling the booking.
# - On the other hand, positive coefficients suggest that an increase in the attribute value increases the probability of customer cancellation.
# - The p-value of a variable indicates its significance. A p-value less than 0.05 (5%) is commonly used as a threshold for statistical significance. Variables with p-values below this threshold are considered significant.
# - However, it's important to note that multicollinearity can affect the coefficients and p-values of variables. Multicollinearity refers to the presence of strong correlations between predictor variables. To obtain reliable coefficients and p-values, it is necessary to address multicollinearity in the data.
# - One approach to detecting multicollinearity is by examining the Variation Inflation Factor (VIF), which measures the extent of multicollinearity between variables. By identifying and addressing multicollinearity, we can improve the reliability of the coefficients and p-values obtained from the model.

# #### Multicollinearity
# - **General Rule of thumb**:
#     - If VIF is between 1 and 5, then there is low multicollinearity.
#     - If VIF is between 5 and 10, we say there is moderate multicollinearity.
#     - If VIF is exceeding 10, it shows signs of high multicollinearity.

# In[142]:


# we will define a function to check VIF
def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif


# In[143]:


checking_vif(X_train)


# Observations:
# - Variables with high VIF values above 5 indicate a strong presence of multicollinearity. In this case, the variables **`market_segment_type_Complementary`**, **`market_segment_type_Corporate`**, **`market_segment_type_Offline`**, and **`market_segment_type_Online`** have VIF values of 4.50, 16.93, 64.12, and 71.18, respectively.
# 
# - Based on the VIF values of the numerical variables alone, all the variables have VIF values below 5, which suggests that there is no significant multicollinearity among these variables. Therefore, there is no immediate need to drop any numerical variables based on their VIF values.
# 
# - Since dummy variables are created from categorical variables, their multicollinearity assessment using VIF may not be appropriate or necessary. Therefore, it is reasonable to ignore the VIF values for the dummy variables in this context.

# #### Dropping high p-value variables
# 
# - We will drop the predictor variables having a p-value greater than 0.05 as they do not significantly impact the target variable.
# - But sometimes p-values change after dropping a variable. So, we'll not drop all variables at once.
# - Instead, we will do the following:
#     - Build a model, check the p-values of the variables, and drop the column with the highest p-value.
#     - Create a new model without the dropped feature, check the p-values of the variables, and drop the column with the highest p-value.
#     - Repeat the above two steps till there are no columns with p-value > 0.05.
# 
# The above process can also be done manually by picking one variable at a time that has a high p-value, dropping it, and building a model again. But that might be a little tedious and using a loop will be more efficient.

# In[144]:


# initial list of columns
cols = X_train.columns.tolist()

# setting an initial max p-value
max_p_value = 1

while len(cols) > 0:
    # defining the train set
    x_train_aux = X_train[cols]

    # fitting the model
    model = sm.Logit(y_train, x_train_aux).fit(disp=False)

    # getting the p-values and the maximum p-value
    p_values = model.pvalues
    max_p_value = max(p_values)

    # name of the variable with maximum p-value
    feature_with_p_max = p_values.idxmax()

    if max_p_value > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols
print(selected_features)


# In[145]:


X_train1 = X_train[selected_features]
X_test1 = X_test[selected_features]


# In[146]:


logit1 = sm.Logit(y_train, X_train1.astype(float))
lg1 = logit1.fit(disp=False)
print(lg1.summary())


# In[147]:


print("Training performance:")
model_performance_classification_statsmodels(lg1, X_train1, y_train)


# Observations:
# 
# - **Now no categorical feature has p-value greater than 0.05, so we'll consider the features in *X_train1* as the final ones and *lg1* as final model.**
# - After dropping the variables with high p-values, the performance on the training data remains closely the same as before.

# ### Coefficient Interpretations

# - Coefficients of **`required_car_parking_space`**, **`arrival_month`**, **`no_of_previous_bookings_not_canceled`**, **`no_of_special_requests`**, **`repeated_guest`** and some others are negative, an increase in these will lead to a decrease in chances of a customer canceling their booking.
# - Coefficients of **`no_of_adults`**, **`no_of_children`**, **`no_of_weekend_nights`**, **`no_of_week_nights`**, **`lead_time`**, **`avg_price_per_room`**, **`type_of_meal_plan_Not Selected`** and some others are positive, an increase in these will lead to a increase in the chances of a customer canceling their booking.

# **Converting coefficients to odds**
# 
# * The coefficients ($\beta$s) of the logistic regression model are in terms of $log(odds)$ and to find the odds, we have to take the exponential of the coefficients
# * Therefore, **$odds =  exp(\beta)$**
# * The percentage change in odds is given as $(exp(\beta) - 1) * 100$

# In[148]:


# converting coefficients to odds
odds = np.exp(lg1.params)

# finding the percentage change
perc_change_odds = (np.exp(lg1.params) - 1) * 100

# removing limit from number of columns to display
pd.set_option("display.max_columns", None)

# adding the odds to a dataframe
pd.DataFrame({"Odds": odds, "Change_odd%": perc_change_odds}, index=X_train1.columns).T


# ###Coefficient interpretations

# - **`no_of_adults`**: Holding all other features constant a 1 unit change in the number of adults will increase the odds of a booking getting cancelled by 1.11 times or a 11.49% increase in the odds of a booking getting cancelled.
# - **`no_of_children`**: Holding all other features constant a 1 unit change in the number of children will increase the odds of a booking getting cancelled by 1.17 times or a 16.55% increase in the odds of a booking getting cancelled.
# - **`no_of_weekend_nights`**: Holding all other features constant a 1 unit change in the number of weekend nights a customer stays at the hotel will increase the odds of a booking getting cancelled by 1.11 times or a 11.47% increase in the odds of a booking getting cancelled.
# - **`no_of_week_nights`**: Holding all other features constant a 1 unit change in the number of weeknights a customer stays at the hotel will increase the odds of a booking getting cancelled by 1.04 times or a 4.26% increase in the odds of a booking getting cancelled.
# - **`required_car_parking_space`**: The odds of a customer who requires a car parking space are 0.2 times less than a customer who doesn't require a car parking space or a 79.70% fewer odds of a customer canceling their booking.
# - **`lead_time`**: Holding all other features constant a 1 unit change in the lead time will increase the odds of a booking getting cancelled by 1.01 times or a 1.58% increase in the odds of a booking getting cancelled.
# - **`no_of_special_requests`**: Holding all other features constant a 1 unit change in the number of special requests made by the customer will decrease the odds of a booking getting cancelled by 0.22 times or a 77.0% decrease in the odds of a booking getting cancelled.
# - **`avg_price_per_room`**: Holding all other features constant a 1 unit change in the lead time will increase the odds of a booking getting cancelled by 1.02 times or a 1.94% increase in the odds of a booking getting cancelled.
# - **`no_of_previous_cancellations`**: Holding all other features constant a 1 unit change in the number of previous bookings not canceled will decrease the odds of a booking getting cancelled by 1.26 times or a 25.71% increase in the odds of a booking getting canceled.
# - **`type_of_meal_plan_Not Selected`**: The odds of a customer who has not selected any meal plan cancelling the booking are 1.33 times more than a customer who has selected a meal plan or a 33.10% higher odds of a booking getting cancelled if a meal plan is not selected. [keeping all the other meal plan types as reference]
# 
# **Interpretation for other attributes can be done similarly.**

# #### Checking model performance on the training set

# In[149]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_train1, y_train)


# In[150]:


print("Training performance:")
log_reg_model_train_perf = model_performance_classification_statsmodels(
    lg1, X_train1, y_train
)
log_reg_model_train_perf


# #### ROC-AUC
# * ROC-AUC on training set

# In[151]:


logit_roc_auc_train = roc_auc_score(y_train, lg1.predict(X_train1))
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# - Logistic Regression model is giving a generalized performance on training and test set.
# - ROC-AUC score of 0.86 on training is quite good.

# #### Model Performance Improvement

# * Let's see if the recall score can be improved further, by changing the model threshold using AUC-ROC Curve.

# #### Optimal threshold using AUC-ROC curve

# In[152]:


# Optimal threshold as per AUC-ROC curve
# The optimal cut off would be where tpr is high and fpr is low
fpr, tpr, thresholds = roc_curve(y_train, lg1.predict(X_train1))

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_auc_roc = thresholds[optimal_idx]
print(optimal_threshold_auc_roc)


# In[153]:


# creating confusion matrix
confusion_matrix_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_auc_roc
)


# In[154]:


# checking model performance for this model
log_reg_model_train_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_auc_roc
)
print("Training performance:")
log_reg_model_train_perf_threshold_auc_roc


# - Recall has increased significantly from 0.63267 to 0.79265 as compared to the previous model.
# - As we will decrease the threshold value, Recall will keep on increasing and the Precision will decrease, but this is not right, we need to choose an optimal balance between recall and precision.
# 
# 

# #### Let's use Precision-Recall curve and see if we can find a better threshold

# In[155]:


y_scores = lg1.predict(X_train1)
prec, rec, tre = precision_recall_curve(y_train, y_scores,)


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plt.figure(figsize=(10, 7))
plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()


# - At 0.42 threshold we get a balanced precision and recall.

# In[156]:


# setting the threshold
optimal_threshold_curve = 0.42


# #### Checking model performance on training set

# In[157]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_train1, y_train, threshold=optimal_threshold_curve)


# In[158]:


log_reg_model_train_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_train1, y_train, threshold=optimal_threshold_curve
)
print("Training performance:")
log_reg_model_train_perf_threshold_curve


# - Model performance has improved as compared to our initial model.
# - Model has given a balanced performance in terms of precision and recall.

# #### Let's check the performance on the test set

# **Using model with default threshold**

# In[160]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test)


# In[161]:


log_reg_model_test_perf = model_performance_classification_statsmodels(
    lg1, X_test1, y_test
)

print("Test performance:")
log_reg_model_test_perf


# * ROC curve on test set

# In[162]:


logit_roc_auc_train = roc_auc_score(y_test, lg1.predict(X_test1))
fpr, tpr, thresholds = roc_curve(y_test, lg1.predict(X_test1))
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Logistic Regression (area = %0.2f)" % logit_roc_auc_train)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# **Using model with threshold=0.37**

# In[163]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test, threshold=optimal_threshold_auc_roc)


# In[164]:


# checking model performance for this model
log_reg_model_test_perf_threshold_auc_roc = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_auc_roc
)
print("Test performance:")
log_reg_model_test_perf_threshold_auc_roc


# **Using model with threshold = 0.42**

# In[165]:


# creating confusion matrix
confusion_matrix_statsmodels(lg1, X_test1, y_test, threshold=optimal_threshold_curve)


# In[166]:


log_reg_model_test_perf_threshold_curve = model_performance_classification_statsmodels(
    lg1, X_test1, y_test, threshold=optimal_threshold_curve
)
print("Test performance:")
log_reg_model_test_perf_threshold_curve


# #### Model performance summary

# In[167]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        log_reg_model_train_perf.T,
        log_reg_model_train_perf_threshold_auc_roc.T,
        log_reg_model_train_perf_threshold_curve.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Logistic Regression-default Threshold",
    "Logistic Regression-0.37 Threshold",
    "Logistic Regression-0.42 Threshold",
]

print("Training performance comparison:")
models_train_comp_df


# In[169]:


# test performance comparison

models_test_comp_df = pd.concat(
    [
        log_reg_model_test_perf.T,
        log_reg_model_test_perf_threshold_auc_roc.T,
        log_reg_model_test_perf_threshold_curve.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Logistic Regression statsmodel",
    "Logistic Regression-0.37 Threshold",
    "Logistic Regression-0.42 Threshold",
]

print("Test performance comparison:")
models_test_comp_df


# ###Observations from Logistic Regression model
# - We have been able to build a predictive model that can be used by the hotel to predict which bookings are likely to be cancelled with an F1 score of 0.68 on the training set and formulate marketing policies accordingly. The logistic regression models are giving a generalized performance on training and test set.
# 
# - **Using the model with default threshold the model will give the lowest recall(~ 0.63) but good precision score (~ 0.73)** - The hotel will be able to predict which bookings will not be cancelled and will be able to provide satisfactory services to those customers which help in maintaining the brand equity but will lose on resources.
# 
# - **Using the model with a 0.37 threshold the model will give a high recall (~ 0.74) but low precision score (~ 0.67)** - The hotel will be able to save resources by correctly predicting the bookings which are likely to be cancelled but might damage the brand equity.
# 
# - **Using the model with a 0.42 threshold the model will give a balance recall (~ 0.70) and precision score (~ 0.70)** - The hotel will be able to maintain a balance between resources and brand equity.

# ### Decision Tree

# #### Data Preparation for modeling (Decision Tree)

# - We want to predict which bookings will be canceled.
# - Before we proceed to build a model, we'll have to encode categorical features.
# - We'll split the data into train and test to be able to evaluate the model that we build on the train data.

# In[170]:


X = data.drop(["booking_status"], axis=1)
Y = data["booking_status"]

X = pd.get_dummies(X, drop_first=True)

# Splitting data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1
)


# In[171]:


print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))


# #### First, let's create functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.
# * The model_performance_classification_sklearn function will be used to check the model performance of models.
# * The confusion_matrix_sklearnfunction will be used to plot the confusion matrix.

# In[172]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# In[173]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# #### Building Decision Tree Model

# In[174]:


model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)


# #### Checking model performance on training set

# In[175]:


confusion_matrix_statsmodels(model, X_train, y_train)


# In[176]:


decision_tree_perf_train = model_performance_classification_sklearn(
    model, X_train, y_train
)
decision_tree_perf_train


# - Accuracy: The accuracy of the model on the training data is 0.99421, which means that it correctly predicted the class labels for approximately 99.421% of the training instances.
# 
# - Recall: The model achieved a recall of 0.98661, which means it correctly identified 98.661% of the positive instances in the training data.
# 
# - Precision: With a precision of 0.99578, the model correctly classified approximately 99.578% of the instances it predicted as positive.
# 
# - F1 Score: The F1 score is 0.99117, which indicates a good trade-off between precision and recall.
# 
# Overall, the summary suggests that the decision tree model performs very well on the training data, with high accuracy, recall, precision, and F1 score.
# 
# Let's check the performance on test data to see if the model is overfitting.

# #### Checking model performance on test set

# In[180]:


confusion_matrix_statsmodels(model, X_test, y_test)


# In[177]:


decision_tree_perf_test = model_performance_classification_sklearn(
    model, X_test, y_test
)
decision_tree_perf_test


# - Comparing these metrics with the ones we obtained for the training set, we can observe a drop in performance across all metrics. This discrepancy suggests that the decision tree model may be overfitting and not able to generalize well on the test set.
# - We will have to prune the decision tree.

# **Before pruning the tree let's check the important features.**

# In[181]:


feature_names = list(X_train.columns)
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# - **`lead_time`** is the most important feature followed by **`avg_price_per_room`**.
# - Now let's prune the tree to see if we can reduce the complexity.

# #### Pruning the tree

# **Pre-Pruning**

# In[182]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1, class_weight="balanced")

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(2, 7, 2),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(f1_score)

# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, scoring=acc_scorer, cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### Checking performance on training set

# In[183]:


confusion_matrix_sklearn(estimator, X_train, y_train)


# In[184]:


decision_tree_tune_perf_train = model_performance_classification_sklearn(
    estimator, X_train, y_train
)
decision_tree_tune_perf_train


# #### Checking performance on test set

# In[185]:


confusion_matrix_sklearn(estimator, X_test, y_test)


# In[186]:


decision_tree_tune_perf_test = model_performance_classification_sklearn(
    estimator, X_test, y_test
)
decision_tree_tune_perf_test


# #### Visualizing the Decision Tree

# In[187]:


plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[188]:


# Text report showing the rules of a decision tree -
print(tree.export_text(estimator, feature_names=feature_names, show_weights=True))


# In[189]:


# importance of features in the tree building

importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ###Observations from decision tree:
# The decision tree model has undergone simplification, resulting in readable and interpretable rules within the tree.
# 
# Furthermore, the model's performance has become more generalized, as it demonstrates improved performance on unseen data.
# 
# Upon analysis, the most important features for the model's predictions are:
# 
# - **`lead_time`**: This feature has a significant impact on the model's predictions, indicating that the amount of time between booking and arrival plays a crucial role in determining the outcome. Here's a breakdown of the rules inferred from the decision tree:
# 
#   1. Bookings made **more than 151 days** before the date of arrival:
#    - If the average price per room is greater than 100 euros and the arrival month is December, then the booking is less likely to be canceled.
#    - If the average price per room is less than or equal to 100 euros and the number of special requests is 0, then the booking is likely to be canceled.
# 
#   2. Bookings made **under 151 days** before the date of arrival:
#    - If a customer has at least 1 special request, the booking is less likely to be canceled.
#    - If the customer didn't make any special requests and the booking was done online, it is more likely to get canceled. If the booking was not done online, it is less likely to be canceled.
# 
# - **`market_segment_type_Online`**: This feature suggests that online bookings made through specific market segments have a strong influence on the model's predictions.
# 
# - **`no_of_special_requests`**: The number of special requests made by guests emerges as an important factor, implying that additional requirements or preferences contribute to the model's decision-making process.
# 
# - **`avg_price_per_room`**: The average price per room is identified as an influential feature, implying that pricing information has a notable impact on the model's predictions.
# 
# Overall, these insights provide valuable information about the decision-making process of the model and highlight the most influential features for predicting the outcome.
# 
# **If we want more complex then we can go in more depth of the tree.**

# **Cost Complexity Pruning**

# In[190]:


clf = DecisionTreeClassifier(random_state=1, class_weight="balanced")
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = abs(path.ccp_alphas), path.impurities


# In[191]:


pd.DataFrame(path)


# In[192]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


# Next, we train a decision tree using effective alphas. The last value
# in ``ccp_alphas`` is the alpha value that prunes the whole tree,
# leaving the tree, ``clfs[-1]``, with one node.

# In[195]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=1, ccp_alpha=ccp_alpha, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


# In[196]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


# #### F1 Score vs alpha for training and testing sets

# In[197]:


f1_train = []
for clf in clfs:
    pred_train = clf.predict(X_train)
    values_train = f1_score(y_train, pred_train)
    f1_train.append(values_train)

f1_test = []
for clf in clfs:
    pred_test = clf.predict(X_test)
    values_test = f1_score(y_test, pred_test)
    f1_test.append(values_test)


# In[198]:


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlabel("alpha")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score vs alpha for training and testing sets")
ax.plot(ccp_alphas, f1_train, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, f1_test, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


# In[199]:


index_best_model = np.argmax(f1_test)
best_model = clfs[index_best_model]
print(best_model)


# #### Checking performance on training set

# In[200]:


confusion_matrix_sklearn(best_model, X_train, y_train)


# In[201]:


decision_tree_post_perf_train = model_performance_classification_sklearn(
    best_model, X_train, y_train
)
decision_tree_post_perf_train


# #### Checking performance on test set

# In[202]:


confusion_matrix_sklearn(best_model, X_test, y_test)


# In[203]:


decision_tree_post_test = model_performance_classification_sklearn(
    best_model, X_test, y_test
)
decision_tree_post_test


# Observations
# 
# - After post pruning the decision tree the performance has generalized on training and test set.
# - We are getting high recall (~ 0.86) with this model but difference between recall and precision(~ 0.77) has increased.

# In[204]:


plt.figure(figsize=(20, 10))

out = tree.plot_tree(
    best_model,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# In[205]:


# Text report showing the rules of a decision tree -

print(tree.export_text(best_model, feature_names=feature_names, show_weights=True))


# In[206]:


importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# Observations from tree
# 
# - The decision tree is significantly more complex compared to the pre-pruned tree.
# - The feature importance remains consistent with the results obtained from the pre-pruned tree.

# #### Comparing Decision Tree models

# In[207]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        decision_tree_perf_train.T,
        decision_tree_tune_perf_train.T,
        decision_tree_post_perf_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Training performance comparison:")
models_train_comp_df


# In[208]:


# testing performance comparison

models_test_comp_df = pd.concat(
    [
        decision_tree_perf_test.T,
        decision_tree_tune_perf_test.T,
        decision_tree_post_test.T,
    ],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree sklearn",
    "Decision Tree (Pre-Pruning)",
    "Decision Tree (Post-Pruning)",
]
print("Test set performance comparison:")
models_test_comp_df


# Observations
# 
# - The default decision tree model is suffering from overfitting on the training data, leading to poor generalization.
# - The pre-pruned tree model demonstrates a balanced performance with respectable values for both precision and recall.
# - Although the post-pruned tree model achieves a high F1 score compared to other models, there is a significant disparity between precision and recall.
# - By utilizing **the pre-pruned decision tree model**, the hotel can effectively manage resources while preserving brand equity.

# ## Actionable Insights and Recommendations
# 
# 1. The **`lead_time`** and the **`no_of_special_requests`** made by the customer play a key role in identifying if a booking will be cancelled or not. Bookings where a customer has made a special request and the booking was done under 151 days to the date of arrival are less likely to be canceled. Based on the analysis, we have the recommendations below:
# 
#   - Prompt Special Request Confirmation:
#     - Implement a system to send automated notifications or emails to customers who have made special requests.
#     - Encourage customers to confirm their special requests promptly to ensure their needs are met.
#     - This proactive approach will help solidify the booking and reduce the likelihood of cancellations.
# 
#   - Emphasize Early Bookings:
#     - Promote early bookings to guests by offering incentives or discounts for reservations made well in advance.
#     - Highlight the benefits of securing bookings under 151 days before the arrival date.
#     - By encouraging customers to book early, the hotel can increase the chances of lower cancellation rates.
# 
#   - Personalize Communication:
#     - Develop a communication strategy that targets customers who have made special requests and booked within the preferred lead time.
#     - Tailor messages to these guests, highlighting their special requests and emphasizing the hotel's commitment to fulfilling their needs.
#     - This personalized approach can create a stronger bond with guests and decrease the likelihood of cancellations.
# 
#   - Streamline Special Request Management:
#     - Implement a robust system for managing and fulfilling special requests efficiently.
#     - Ensure that staff members are well-trained and equipped to handle various requests.
#     - By effectively managing special requests, the hotel can enhance guest satisfaction and reduce the chances of cancellations.
# 
#   - Continuously Monitor and Analyze Data:
#     - Regularly review and analyze booking data, lead time, and special requests to identify any evolving trends or patterns.
#     - Stay informed about changes in customer preferences and adapt strategies accordingly.
#     - This proactive approach will enable the hotel to stay ahead of potential cancellation risks and optimize its operations.
# 
# 2. For the refund policy, the hotel can implement more stringent cancellation policies to address specific situations. For bookings with a high average price per room and associated special requests, a full refund may not be granted due to the significant loss of resources involved. Ideally, it is desirable to have consistent cancellation policies across all market segments. However, our analysis reveals a high percentage of online bookings being canceled. Therefore, online cancellations should result in a lower percentage of refund for customers.
# 
# 3. According to our analysis, bookings with a total length of stay exceeding 5 days showed a higher probability of being canceled. Based on the analysis, here are two suggestions for the hotel regarding bookings with a total length of stay exceeding 5 days, which have a higher likelihood of being canceled:
# 
#   - Flexible Modification Option: Offer a flexible modification option for bookings with a longer duration. This allows guests to make changes to their reservation without incurring additional charges or penalties. By providing this option, guests may be more inclined to modify their plans instead of canceling altogether.
# 
#   - Discounted Non-Refundable Rates: Introduce discounted non-refundable rates specifically for bookings with a total length of stay exceeding 5 days. These rates would offer a significant discount compared to standard rates but would be non-refundable. This approach can encourage guests to commit to their longer stays while enjoying cost savings, reducing the likelihood of cancellations.
# 
# 4. During December and January, cancellation rate is comparatively low. This trend could be attributed to various factors such as holiday season plans, family gatherings, or reduced business travel during this period. Therefore, the hotel can consider the meausres below:
#   - Promote special offers and run targeted marketing campaigns during the months of December and January to leverage the lower cancellation rates and attract more guests during this period.
#   - Provide flexible booking options and incentives, such as discounted rates or exclusive perks, to encourage guests to book longer stays or make advanced reservations, thereby reducing the likelihood of cancellations and maximizing revenue potential.
# 
# 5. October and September saw the highest number of bookings but also high number of cancellations. Therefore, the hotel can consider the measure below:
#   - Analyze the reasons behind the high number of cancellations during October and September. Identify any patterns or common factors contributing to cancellations, such as seasonal changes, events, or other external factors. Based on the findings, develop targeted strategies to address these specific concerns and mitigate cancellations during these months.
# 
#   - Implement a proactive communication strategy with guests who have booked during October and September. Reach out to them prior to their arrival to confirm their reservations, provide personalized recommendations or incentives to encourage them to maintain their bookings, and address any concerns they may have. By establishing a direct line of communication and demonstrating attentive customer service, you can increase guest confidence and reduce the likelihood of cancellations.
# 
# 6. Our analysis shows that there are very few repeated customers and the cancellation among them is very less. Therefore, the hotel can:
#   - Focus on customer retention strategies: Implement initiatives to encourage repeat bookings and enhance customer loyalty. Offer special incentives or rewards programs for returning customers, such as discounts, exclusive perks, or personalized offers. Provide exceptional customer service to create positive experiences that will increase the likelihood of repeat visits and reduce cancellations.
# 
#   - Enhance customer satisfaction and engagement: Conduct surveys or gather feedback from both repeat and non-repeat customers to identify areas for improvement. Use this information to enhance the overall guest experience, address any pain points, and exceed customer expectations. By consistently delivering exceptional service and meeting customer needs, you can build long-term relationships, increase customer loyalty, and minimize cancellations.
# 
# By implementing these suggestions, the hotel can enhance its booking process, reduce cancellations, and provide a superior guest experience.
