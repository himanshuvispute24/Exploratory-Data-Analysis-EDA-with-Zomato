
                          #######Exploratory Data Analysis (EDA) with Zomato########



import numpy as np
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import random
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # Other options: "Qt5Agg", "Agg", etc.

#Data load
data = pd.read_csv("zomato_restaurants_in_India.csv")
print (data.head(10))
print(data.tail())

print(data.city.nunique())

print(data.city.unique())

mumbai_data = data[data["city"] == "Mumbai"]
print(mumbai_data)

New_Delhi=data[data["city"]=="New Delhi"]
print(New_Delhi)

Pune_data=data[data["city"]=="Pune"]
print(Pune_data)

print(data.shape)

print(data.info())

filtered_data = data[data["average_cost_for_two"] == 30000]
print(filtered_data)

summary = data.describe()
print(summary)

print(data.head())

# Shape before removing duplicates
print("Shape before removing duplicates:", data.shape)

# Remove duplicates based on 'res_id'
data.drop_duplicates(["res_id"], keep='first', inplace=True)

# Check the shape after removing duplicates
print("Shape after removing duplicates:", data.shape)

missing_values = data.isnull().sum()
print(missing_values)

# unique_establishments
unique_establishments = data["establishment"].unique()
print(unique_establishments)

print(data["establishment"].unique()[0])
print(type(data["establishment"].unique()[0]))

# Removing [' '] from each value
print(data["establishment"].unique()[0])
data["establishment"] = data["establishment"].apply(lambda x:x[2:-2])
print(data["establishment"].unique()[0])

# Changing ''  to 'NA'
print(data["establishment"].unique())
data["establishment"] = data["establishment"].apply(lambda x : np.where(x=="", "NA", x))
print(data["establishment"].unique())

x = 10
y = 11
print("New x:", x)  # Output: New x: 10
print("New y:", y)  # Output: New y: 11

result = x == y
print(result)

shimla_count = len(data[data["city"] == "Shimla"])
print("Numbers of city is Shimla:", shimla_count)

Agra_count =len(data[data["city"]=="Agra"])
print("Numbers of city is Agra:", Agra_count)

unique_cities_count = len(data["city"].unique())
print("Number of unique cities:", unique_cities_count)

unique_cities = data["city"].unique()
print(unique_cities)

jabalpur_data = data[data["city"] == "Jabalpur"]
print(jabalpur_data)

unique_localities = data["locality"].nunique()
print("Number of unique localities:", unique_localities)

unique_country_ids = data["country_id"].unique()
print(unique_country_ids)

unique_locality_verbose = data["locality_verbose"].nunique()
print("Number of unique locality_verbose values:", unique_locality_verbose)

print("Number of unique cuisines:", data["cuisines"].nunique())
print("Unique cuisines:", data["cuisines"].unique())

data["cuisines"] = data["cuisines"].fillna("No cuisine")

# Check the result
print(data["cuisines"].head())

# Initialize an empty list to store individual cuisines
cuisines = []

# Split combined cuisines and extend the list
data["cuisines"].apply(lambda x: cuisines.extend(x.split(", ")))

# Convert the list to a pandas Series
cuisines = pd.Series(cuisines)

# Count the number of unique cuisines
print("Total number of unique cuisines =", cuisines.nunique())

print("Number of unique timings:", data["timings"].nunique())
print("Unique timings:", data["timings"].unique())

unique_average_costs = data["average_cost_for_two"].nunique()
print("Number of unique average costs for two:", unique_average_costs)

unique_price_ranges = data["price_range"].unique()
print(unique_price_ranges)

unique_currency=data["currency"].unique()
print(unique_currency)

print(data["highlights"].nunique())
print(data["highlights"].unique())

# Initialize empty list
hl = []

# Extract highlights
data["highlights"].apply(lambda x: hl.extend(x[2:-2].split("', '")))

# Convert to pandas Series
hl = pd.Series(hl)

# Print number of unique highlights
print("Total number of unique highlights =", hl.nunique())

# Calculate mean, min, and max
result = data[["aggregate_rating", "votes", "photo_count"]].describe().loc[["mean", "min", "max"]]

# Print result
print(result)

# Get unique values
unique_values = data["opentable_support"].unique()

# Print unique values
print(unique_values)

# Get unique values
unique_values = data["delivery"].unique()

# Print unique values
print(unique_values)


# Get unique values
unique_values = data["takeaway"].unique()

# Print unique values
print(unique_values)

#Exploratory Data Analysis (EDA)¶
#Restaurant chains
#Here chains represent restaurants with more than one outlet

##Chains vs Outlets

# Count occurrences of each unique value
outlets = data["name"].value_counts()

# Print the result
print(outlets)

# Filter the outlets
chains = outlets[outlets >= 2]
single = outlets[outlets == 1]

print("Chains:", chains)
print("Singles:", single)

print(data.shape)
print(chains)

# Count occurrences of each unique restaurant name
outlets = data["name"].value_counts()

# Separate single-outlet restaurants and chain restaurants
single = outlets[outlets == 1]  # Restaurants that are not part of a chain
chains = outlets[outlets >= 2]  # Restaurants that are part of a chain

# Print the total restaurants and statistics
print("Total Restaurants = ", data.shape[0])
print("Total Restaurants that are part of some chain = ", data.shape[0] - single.shape[0])
print("Percentage of Restaurants that are part of a chain = ", np.round((data.shape[0] - single.shape[0]) / data.shape[0], 2) * 100, "%")

# Count occurrences of each unique restaurant name
outlets = data["name"].value_counts()

# Filter to include only chain restaurants (2 or more outlets)
chains = outlets[outlets >= 2]

# Display the top 10 chains
print(chains.head(10))

top10_chains = data["name"].value_counts().sort_values(ascending=False).head(10).sort_values(ascending=True)
print(top10_chains)


# Example data
top10_chains = pd.Series({
"Domino's Pizza":     399,
"Cafe Coffee Day":    315,
"KFC":                204,
"Baskin Robbins":     202,
"Keventers":          189,
"Subway":             178,
"McDonald's":         130,
"Pizza Hut":          125,
"Barbeque Nation":    112,
"Burger King":        110
})

# Data preparation
height = top10_chains.values
bars = top10_chains.index
y_pos = np.arange(len(bars))

# Figure and axes setup
fig = plt.figure(figsize=[11, 7], frameon=False)
ax = fig.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Custom color palette
colors = ["#f9cdac", "#f2a49f", "#ec7c92", "#e65586", "#bc438b", "#933291", "#692398", "#551c7b", "#41155e", "#2d0f41"]

# Create horizontal bar plot
plt.barh(y_pos, height, color=colors)

# Customize labels and ticks
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of outlets in India")

# Add value labels
for i, v in enumerate(height):
    ax.text(v + 3, i, str(v), color='#424242')

# Top 10 Restaurant chain in India (by number of outlets
plt.title("Top 10 Restaurant chain in India (by number of outlets)")

# Display plot
plt.show()

top_5 = data["name"].value_counts().head()
print(top_5)

atleast_5_outlets = outlets[outlets > 4]
print(atleast_5_outlets)

top10_chains2 = (
    data[data["name"].isin(atleast_5_outlets.index)]
    .groupby("name")["aggregate_rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values(ascending=True)  # Sort for better visualization
)

print(top10_chains2)

# Convert values to a rounded Series
height = pd.Series(top10_chains2.values).map(lambda x: np.round(x, 2))
bars = top10_chains2.index
y_pos = np.arange(len(bars))

# Create figure and axis
fig, ax = plt.subplots(figsize=(11, 7))

# Customize spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Color scheme
colors = ['#fded86', '#fce36b', '#f7c65d', '#f1a84f', '#ec8c41', '#e76f34',
          '#e25328', '#b04829', '#7e3e2b', '#4c3430']

# Plot horizontal bar chart
plt.barh(y_pos, height, color=colors)

# Formatting axes and labels
plt.xlim(3)  # Set x-axis minimum limit
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Average Rating", fontsize=12, color="#424242")
plt.title("Top 10 Restaurant Chains in India (by Average Rating)", fontsize=14, color="#424242")

# Add text labels on bars
for i, v in enumerate(height):
    ax.text(v + 0.01, i, str(v), color='#424242', va='center', fontsize=10)

# Show plot
plt.show()



#Establishment Types
#Number of restaurants (by establishment type)

# Compute the top 5 establishment types by count
est_count = (
    data.groupby("establishment")["res_id"]
    .count()
    .sort_values(ascending=False)
    .head(5)
)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Customize spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Define colors
colors = ["#2d0f41", "#933291", "#e65586", "#f2a49f", "#f9cdac"]

# Plot bar chart
plt.bar(est_count.index, est_count.values, color=colors)

# Format x and y ticks
plt.xticks(rotation=45, color="#424242")  # Rotating to avoid overlap
plt.yticks(np.arange(0, max(est_count.values) + 5000, 5000), color="#424242")
plt.xlabel("Top 5 Establishment Types", fontsize=12, color="#424242")
plt.ylabel("Number of Restaurants", fontsize=12, color="#424242")

# Add text labels on bars
for i, (est, v) in enumerate(zip(est_count.index, est_count.values)):
    ax.text(i, v + 500, str(v), color='#424242', ha='center', fontsize=10)

# Set title
plt.title("Number of Restaurants by Establishment Type", fontsize=14, color="#424242")

# Show plot
plt.show()

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'establishment', calculate the mean of 'aggregate_rating', and sort
rating_by_est = data.groupby("establishment")["aggregate_rating"].mean().sort_values(ascending=False)[:10]

# Display the results
print(rating_by_est)

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'establishment', calculate the mean of 'votes', and sort
votes_by_est = data.groupby("establishment")["votes"].mean().sort_values(ascending=False)[:10]

# Display the results
print(votes_by_est)

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'establishment', calculate the mean of 'photo_count', and sort
photo_count_by_est = data.groupby("establishment")["photo_count"].mean().sort_values(ascending=False)[:10]

# Display the results
print(photo_count_by_est)



#Cities
#Number of restaurants (by city)

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'city', count the number of restaurant IDs, and sort
city_counts = data.groupby("city").count()["res_id"].sort_values(ascending=True)[-10:]

# Prepare data for plotting
height = pd.Series(city_counts.values)
bars = city_counts.index
y_pos = np.arange(len(bars))

# Create the figure and axis
fig = plt.figure(figsize=[11, 7], frameon=False)
ax = fig.gca()

# Customize the spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Define colors for the bars
colors = ['#dcecc9', '#aadacc', '#78c6d0', '#48b3d3', '#3e94c0',
          '#3474ac', '#2a5599', '#203686', '#18216b', '#11174b']

# Create the horizontal bar chart
plt.barh(y_pos, height, color=colors)

# Set limits and ticks
plt.xlim(0, height.max() + 20)  # Adjust xlim to fit the text labels
plt.xticks(color="#424242")
plt.yticks(y_pos, bars, color="#424242")
plt.xlabel("Number of Outlets")

# Add data labels to the bars
for i, v in enumerate(height):
    ax.text(v + 1, i, str(v), color='#424242')  # Adjusted for better spacing

# Set the title
plt.title("Number of Restaurants (by City)")

# Show the plot
plt.show()

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'city', calculate the mean of 'aggregate_rating', and sort
rating_by_city = data.groupby("city")["aggregate_rating"].mean().sort_values(ascending=False)[:10]

# Display the results
print(rating_by_city)


# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'city', calculate the mean of 'votes', and sort
votes_by_city = data.groupby("city")["votes"].mean().sort_values(ascending=False)[:10]

# Display the results
print(votes_by_city)

# Assuming 'data' is your DataFrame containing the relevant information
# Group by 'city', calculate the mean of 'photo_count', and sort
photo_count_by_city = data.groupby("city")["photo_count"].mean().sort_values(ascending=False)[:10]

# Display the results
print(photo_count_by_city)



#Cuisine
#Unique cuisines

# Assuming 'data' is your DataFrame and 'cuisines' is a column in that DataFrame
# For example, if 'cuisines' is a column in 'data':
cuisines = data['cuisines']  # Replace 'cuisines' with the actual column name if different

# Print the total number of unique cuisines
print("Total number of unique cuisines =", cuisines.nunique())

#number of restaurants (by cuisine)
# Assuming 'data' is your DataFrame and 'cuisines' is a column in that DataFrame
# For example, if 'cuisines' is a column in 'data':
cuisines = data['cuisines']  # Replace 'cuisines' with the actual column name if different

# Count the occurrences of each cuisine and get the top 5
c_count = cuisines.value_counts()[:5]

# Create the figure
fig = plt.figure(figsize=[8, 5], frameon=False)

# Get the current axis
ax = fig.gca()

# Customize the spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Define colors for the bars
colors = ['#4c3430', '#b04829', '#ec8c41', '#f7c65d', '#fded86']

# Create the bar chart
plt.bar(c_count.index, c_count.values, color=colors)

# Set x and y ticks
plt.xticks(rotation=45, color="#424242")  # Rotate x labels for better visibility
plt.yticks(range(0, max(c_count.values) + 5000, 5000), color="#424242")
plt.xlabel("Top 5 Cuisines")

# Add data labels on top of the bars
for i, v in enumerate(c_count):
    ax.text(i, v + 500, str(v), color='#424242', ha='center')  # Center the text above the bars

# Set the title
plt.title("Number of Restaurants (by Cuisine Type)")

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()

#Highest rated cuisines
# Assuming 'data' is your DataFrame containing the relevant information
# Step 1: Split the 'cuisines' column into lists
data["cuisines2"] = data['cuisines'].apply(lambda x: x.split(", "))

# Step 2: Create a list of unique cuisines
cuisines_list = data['cuisines'].str.split(', ').explode().unique().tolist()

# Step 3: Initialize a DataFrame to hold the sum and total count of restaurants for each cuisine
zeros = np.zeros(shape=(len(cuisines_list), 2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum", "Total"])

# Step 4: Calculate the sum and total for each cuisine
for index, row in data.iterrows():
    for cuisine in row['cuisines2']:
        c_and_r.at[cuisine, "Total"] += 1  # Increment the total count
        c_and_r.at[cuisine, "Sum"] += row['aggregate_rating']  # Assuming you want to sum ratings

# Step 5: Optionally, calculate the average rating for each cuisine
c_and_r['Average Rating'] = c_and_r['Sum'] / c_and_r['Total']
c_and_r = c_and_r.fillna(0)  # Fill NaN values with 0 if there are cuisines with no ratings

# Display the resulting DataFrame
print(c_and_r)

# Assuming 'data' is your DataFrame containing the relevant information
# Step 1: Split the 'cuisines' column into lists
data["cuisines2"] = data['cuisines'].apply(lambda x: x.split(", "))

# Step 2: Create a list of unique cuisines
cuisines_list = data['cuisines'].str.split(', ').explode().unique().tolist()

# Step 3: Initialize a DataFrame to hold the sum and total count of restaurants for each cuisine
zeros = np.zeros(shape=(len(cuisines_list), 2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum", "Total"])

# Step 4: Calculate the sum and total for each cuisine
for i, x in data.iterrows():
    for j in x["cuisines2"]:
        c_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        c_and_r.loc[j, "Total"] += 1  # Increment the total count

# Step 5: Optionally, calculate the average rating for each cuisine
c_and_r['Average Rating'] = c_and_r['Sum'] / c_and_r['Total']
c_and_r = c_and_r.fillna(0)  # Fill NaN values with 0 if there are cuisines with no ratings

# Display the resulting DataFrame
print(c_and_r)

#
# Step 1: Split the 'cuisines' column into lists
data["cuisines2"] = data['cuisines'].apply(lambda x: x.split(", "))

# Step 2: Create a list of unique cuisines
cuisines_list = data['cuisines'].str.split(', ').explode().unique().tolist()

# Step 3: Initialize a DataFrame to hold the sum and total count of restaurants for each cuisine
zeros = np.zeros(shape=(len(cuisines_list), 2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum", "Total"])

# Step 4: Calculate the sum and total for each cuisine
for i, x in data.iterrows():
    for j in x["cuisines2"]:
        c_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        c_and_r.loc[j, "Total"] += 1  # Increment the total count

# Step 5: Calculate the mean rating for each cuisine
c_and_r["Mean"] = c_and_r["Sum"] / c_and_r["Total"]

# Step 6: Fill NaN values with 0 (in case there are cuisines with no ratings)
c_and_r = c_and_r.fillna(0)

# Display the resulting DataFrame
print(c_and_r)

#
# Step 1: Split the 'cuisines' column into lists
data["cuisines2"] = data['cuisines'].apply(lambda x: x.split(", "))

# Step 2: Create a list of unique cuisines
cuisines_list = data['cuisines'].str.split(', ').explode().unique().tolist()

# Step 3: Initialize a DataFrame to hold the sum and total count of restaurants for each cuisine
zeros = np.zeros(shape=(len(cuisines_list), 2))
c_and_r = pd.DataFrame(zeros, index=cuisines_list, columns=["Sum", "Total"])

# Step 4: Calculate the sum and total for each cuisine
for i, x in data.iterrows():
    for j in x["cuisines2"]:
        c_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        c_and_r.loc[j, "Total"] += 1  # Increment the total count

# Step 5: Calculate the mean rating for each cuisine
c_and_r["Mean"] = c_and_r["Sum"] / c_and_r["Total"]

# Step 6: Fill NaN values with 0 (in case there are cuisines with no ratings)
c_and_r = c_and_r.fillna(0)

# Step 7: Sort by Mean and select the top 10 cuisines
top_cuisines = c_and_r[["Mean", "Total"]].sort_values(by="Mean", ascending=False)[:10]

# Display the top cuisines
print(top_cuisines)




#Highlights/Features of restaurants¶
#Unique highlights

#
# Assuming 'hl' is a Series containing the cuisine data
# If 'hl' is meant to be the 'cuisines' column, you can do:
hl = data['cuisines'].str.split(', ').explode()  # Split and explode to get individual cuisines

# Print the total number of unique cuisines
print("Total number of unique cuisines =", hl.nunique())


#Number of restaurants (by highlights)

# Step 1: Explode the highlights into individual entries
hl = data['highlights'].str.split(', ').explode()

# Step 2: Count the occurrences of each highlight and get the top 5
h_count = hl.value_counts()[:5]

# Step 3: Create the figure
fig = plt.figure(figsize=[10, 6], frameon=False)
ax = fig.gca()

# Customize the spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Define colors for the bars
colors = ['#11174b', '#2a5599', '#3e94c0', '#78c6d0', '#dcecc9']

# Create the bar chart
plt.bar(h_count.index, h_count.values, color=colors)

# Set x and y ticks
plt.xticks(rotation=45, color="#424242")  # Rotate x labels for better visibility
plt.yticks(range(0, h_count.max() + 1000, 10000), color="#424242")
plt.xlabel("Top 5 Highlights")

# Add data labels on top of the bars
for i, v in enumerate(h_count):
    ax.text(i, v + 500, str(v), color='#424242', ha='center')  # Center the text above the bars

# Set the title
plt.title("Number of Restaurants (by Highlights)")

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()



#Highest rated highlights

# Access the first element of the 'highlights' column using iloc
first_highlight = data["highlights"].iloc[0]

# Print the result
print(first_highlight)

# Step 1: Clean and split the 'highlights' column
data["highlights2"] = data['highlights'].apply(lambda x: x[2:-2].split("', '"))

# Step 2: Explode the highlights into individual entries
hl = data['highlights2'].explode()

# Step 3: Create a list of unique highlights
hl_list = hl.unique().tolist()

# Step 4: Initialize a DataFrame to hold the sum and total count of restaurants for each highlight
zeros = np.zeros(shape=(len(hl_list), 2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum", "Total"])

# Display the resulting DataFrame
print(h_and_r)

# Step 2: Explode the highlights into individual entries
hl = data['highlights2'].explode()

# Step 3: Create a list of unique highlights
hl_list = hl.unique().tolist()

# Step 4: Initialize a DataFrame to hold the sum and total count of restaurants for each highlight
zeros = np.zeros(shape=(len(hl_list), 2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum", "Total"])

# Step 5: Calculate the sum and total for each highlight
for i, x in data.iterrows():
    for j in x["highlights2"]:
        h_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        h_and_r.loc[j, "Total"] += 1  # Increment the total count

# Display the resulting DataFrame
print(h_and_r)

# Step 1: Clean and split the 'highlights' column
data["highlights2"] = data['highlights'].apply(lambda x: x[2:-2].split("', '"))

# Step 2: Explode the highlights into individual entries
hl = data['highlights2'].explode()

# Step 3: Create a list of unique highlights
hl_list = hl.unique().tolist()

# Step 4: Initialize a DataFrame to hold the sum and total count of restaurants for each highlight
zeros = np.zeros(shape=(len(hl_list), 2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum", "Total"])

# Step 5: Calculate the sum and total for each highlight
for i, x in data.iterrows():
    for j in x["highlights2"]:
        h_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        h_and_r.loc[j, "Total"] += 1  # Increment the total count

# Step 6: Calculate the mean rating for each highlight
h_and_r["Mean"] = h_and_r["Sum"] / h_and_r["Total"]

# Step 7: Fill NaN values with 0 (in case there are highlights with no ratings)
h_and_r = h_and_r.fillna(0)

# Display the resulting DataFrame
print(h_and_r)

# Step 1: Clean and split the 'highlights' column
data["highlights2"] = data['highlights'].apply(lambda x: x[2:-2].split("', '"))

# Step 2: Explode the highlights into individual entries
hl = data['highlights2'].explode()

# Step 3: Create a list of unique highlights
hl_list = hl.unique().tolist()

# Step 4: Initialize a DataFrame to hold the sum and total count of restaurants for each highlight
zeros = np.zeros(shape=(len(hl_list), 2))
h_and_r = pd.DataFrame(zeros, index=hl_list, columns=["Sum", "Total"])

# Step 5: Calculate the sum and total for each highlight
for i, x in data.iterrows():
    for j in x["highlights2"]:
        h_and_r.loc[j, "Sum"] += x["aggregate_rating"]  # Update the sum of ratings
        h_and_r.loc[j, "Total"] += 1  # Increment the total count

# Step 6: Calculate the mean rating for each highlight
h_and_r["Mean"] = h_and_r["Sum"] / h_and_r["Total"]

# Step 7: Fill NaN values with 0 (in case there are highlights with no ratings)
h_and_r = h_and_r.fillna(0)

# Display the resulting DataFrame
print(h_and_r)



#Highlights wordcloud
#Here we will create a wordcloud of top 30 highlights
#
# Step 1: Clean and split the 'highlights' column
data["highlights2"] = data['highlights'].apply(lambda x: x[2:-2].split("', '"))

# Step 2: Explode the highlights into individual entries
hl = data['highlights2'].explode()

# Step 3: Create a string of highlights for the word cloud
hl_str = " ".join(hl)

# Step 4: Generate the word cloud
wordcloud = WordCloud(width=800, height=500,
                      background_color='white',
                      min_font_size=10, max_words=30).generate(hl_str)

# Step 5: Display the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()



#Ratings and cost
#Ratings distribution
#Let's see how the ratings are distributes

# Step 1: Create a KDE plot for the 'aggregate_rating' column
plt.figure(figsize=(10, 6))  # Set the figure size
sns.kdeplot(data['aggregate_rating'], fill=True, color='blue')  # Create the KDE plot
plt.title("Ratings Distribution")  # Set the title
plt.xlabel("Aggregate Rating")  # Set the x-axis label
plt.ylabel("Density")  # Set the y-axis label
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()  # Display the plot


#Avergae cost for two distribution

# Step 1: Create a KDE plot for the 'average_cost_for_two' column
plt.figure(figsize=(10, 6))  # Set the figure size
sns.kdeplot(data['average_cost_for_two'], fill=True, color='blue')  # Create the KDE plot

# Step 2: Set x-axis limits and ticks
plt.xlim([0, 6000])  # Set the x-axis limits
plt.xticks(range(0, 6001, 500))  # Set the x-axis ticks

# Step 3: Set the title and labels
plt.title("Average Cost for Two Distribution")  # Set the title
plt.xlabel("Average Cost for Two")  # Set the x-axis label
plt.ylabel("Density")  # Set the y-axis label

# Step 4: Display the plot
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()  # Show the plot

# Sample data creation (for demonstration purposes)
# Step 1: Create a KDE plot for the 'average_cost_for_two' column
plt.figure(figsize=(10, 6))  # Set the figure size
sns.kdeplot(data['average_cost_for_two'], fill=True, color='blue')  # Create the KDE plot

# Step 2: Set x-axis limits and ticks
plt.xlim([0, 6000])  # Set the x-axis limits
plt.xticks(range(0, 6001, 500))  # Set the x-axis ticks

# Step 3: Set the title and labels
plt.title("Average Cost for Two Distribution")  # Set the title
plt.xlabel("Average Cost for Two")  # Set the x-axis label
plt.ylabel("Density")  # Set the y-axis label

# Step 4: Display the plot
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()  # Show the plot


#Price range count

# Step 1: Count the number of restaurants in each price range
pr_count = data.groupby("price_range").count()["name"]

# Step 2: Create the figure
fig = plt.figure(figsize=[8, 5], frameon=False)
ax = fig.gca()

# Customize the spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#424242")
ax.spines["bottom"].set_color("#424242")

# Define colors for the bars
colors = ["#2d0f41", '#933291', "#f2a49f", "#f9cdac"]

# Step 3: Create the bar chart
plt.bar(pr_count.index, pr_count.values, color=colors)

# Step 4: Set x and y ticks
plt.xticks(range(1, len(pr_count.index) + 1), color="#424242")  # Adjusted to match price range indices
plt.yticks(range(0, max(pr_count.values) + 5000, 5000), color="#424242")
plt.xlabel("Price Ranges")

# Step 5: Add data labels on top of the bars
for i, v in enumerate(pr_count):
    ax.text(i + 1, v + 0.5, str(v), color='#424242', ha='center')  # Adjusted for correct positioning

# Step 6: Set the title
plt.title("Number of Restaurants (by Price Ranges)")

# Step 7: Display the plot
plt.show()



#Relation between Average price for two and Rating

# Step 1: Calculate the correlation matrix
correlation_matrix = data[["average_cost_for_two", "aggregate_rating"]].corr()

# Step 2: Extract the correlation value and round it to 2 decimal places
correlation_value = np.round(correlation_matrix["average_cost_for_two"]["aggregate_rating"], 2)

# Print the result
print("Correlation between average cost for two and aggregate rating:", correlation_value)

#A correlation can be seen between restaurant average cost and rating

plt.plot("average_cost_for_two", "aggregate_rating", data=data, linestyle="none", marker="o", markersize=5, color="blue")

# Set limits for the x-axis
plt.xlim([0, 6000])

# Add title and labels
plt.title("Relationship between Average Cost and Rating")
plt.xlabel("Average Cost for Two")
plt.ylabel("Ratings")

# Display the plot
plt.show()

#There is definitely a direct relation between the two. Let's take a smaller sample to draw a clearer scatter plot.
# Create the scatter plot
plt.plot("average_cost_for_two", "aggregate_rating", data=data, linestyle="none", marker="o", markersize=5, color="blue", alpha=0.5)

# Set limits for the x-axis
plt.xlim([0, 3000])

# Add title and labels
plt.title("Relationship between Average Cost and Rating (Sampled Data)")
plt.xlabel("Average Cost for Two")
plt.ylabel("Ratings")

# Display the plot
plt.show()

#Relation between Price range and Rating

# Calculate the correlation matrix
correlation_matrix = data[["price_range", "aggregate_rating"]].corr()

# Access the correlation value and round it
correlation_value = np.round(correlation_matrix["price_range"]["aggregate_rating"], 2)

print(correlation_value)

# Assuming 'data' is your DataFrame, and it has columns 'price_range' and 'aggregate_rating'
# Create the boxplot
sns.boxplot(x='price_range', y='aggregate_rating', data=data)

# Set the y-axis limit to start from 1
plt.ylim(1, data['aggregate_rating'].max() + 0.5)  # Adjust the upper limit dynamically

# Add title and labels
plt.title("Relationship between Price Range and Ratings")
plt.xlabel("Price Range")
plt.ylabel("Aggregate Rating")

# Display the plot
plt.show()



"""Now, it is clear. The higher the price a restaurant charges, more services they provide 
and hence more chances of getting good ratings from their customers."""



