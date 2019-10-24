import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

nyc_data = pd.read_csv('data/AB_NYC_2019.csv')

# =========== Data Visualization =========== 
# print("nyc shape: ", nyc_data.shape)
# print("nyc column names: ", nyc_data.columns)

nyc_col = nyc_data.columns
nyc_num_rows = nyc_data.shape[0]


# =========== Data Cleaning ===========

del_boston_cols = ["id", "listing_url", "scrape_id", "last_scraped", "name", "summary", "space", "description", "experiences_offered", "neighborhood_overview", "notes", "interaction", "house_rules",
 "thumbnail_url", "medium_url", "picture_url", "xl_picture_url", "host_id", "host_url", "host_name", "host_since", "host_location", "host_about", "host_response_time", "host_response_rate",
  "host_acceptance_rate", "host_is_superhost", "host_thumbnail_url", "host_picture_url", "host_neighbourhood", "host_listings_count", "host_total_listings_count", "host_verifications", "host_has_profile_pic", "host_identity_verified"]
del_nyc_cols = ["id", "name", "host_id", "host_name"]

# Put price in numpy array
nyc_price_feature = nyc_data['price']
nyc_price_feature = nyc_price_feature.to_numpy()

# Remove columns
nyc_data_clean = nyc_data.drop(del_nyc_cols, axis=1)
# nyc_data_clean.head()

# Split continuous and categorical features
nyc_cts_feats = ['latitude', 'longitude', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
nyc_cat_feats = ['neighbourhood_group', 'neighbourhood', 'room_type']

nyc_cts_data = nyc_data_clean[nyc_cts_feats].copy()
nyc_cat_data = nyc_data_clean[nyc_cat_feats].copy()


# =========== New York Continuous Data ===========
nyc_price_feature_df = pd.DataFrame(nyc_price_feature)

# Replace NaN in reviews_per_month with 0
nyc_data_clean['reviews_per_month'] = nyc_data_clean['reviews_per_month'].fillna(0)

# Get correlation of of price vs continous data
for feature in nyc_cts_data:
  corr, _ = pearsonr(nyc_price_feature, nyc_data_clean[feature])
  print('Pearsons correlation of ' + feature + ': %.3f' % corr)

  # Plot correlation between price and feature
  concat_df = pd.concat([nyc_price_feature_df, nyc_data_clean[feature]], ignore_index=True, axis=1)
  concat_df.columns = ['price', feature]
  # sns.pairplot(concat_df)

  # Pair plot of every continous feature vs each other, too long to run
  # nyc_price_feature_df = pd.concat([nyc_price_feature_df, nyc_data_clean[feature]], ignore_index=True, axis=1)

# Pair plot of every continous feature vs each other, too long to run
# nyc_price_feature_df.columns = ['price'] + nyc_cts_feats
# sns.pairplot(nyc_price_feature_df)


# =========== New York Categorical Data ===========

# Print the mean price by each categorical data, sorted from highest to lowest. 
# Store sorted lists of each categorical feature in nyc_cat_price
nyc_cat_price = []
for feature in nyc_cat_feats:
  print("\n" + feature)
  list_feats = []
  for unique in nyc_data_clean[feature].unique():
    list_feats.append([nyc_data[nyc_data[feature] == unique]['price'].mean(), unique])
  
  list_feats.sort(reverse=True)
  nyc_cat_price.append(list_feats)

  for i in list_feats:
    print(i[1], "%.2f" % i[0])

# Create new dataframes of price and each categorical feature and plot
nyc_cat_price_df = pd.DataFrame(nyc_cat_price[1])
nyc_cat_price_df.columns = ['price', 'neighbourhood']
nyc_cat_price_df2 = pd.DataFrame(nyc_cat_price[0])
nyc_cat_price_df2.columns = ['price', 'neighbourhood group']
nyc_cat_price_df3 = pd.DataFrame(nyc_cat_price[2])
nyc_cat_price_df3.columns = ['price', 'room type']

# Plot price vs each categorical data
for df in [nyc_cat_price_df, nyc_cat_price_df2, nyc_cat_price_df3]:
  plt.figure(figsize = (6, 6))
  bar = sns.barplot(x = df.columns[1], y = 'price', data=df)
  xt = plt.xticks(rotation=90)
  fig = bar.get_figure()

# Top 10 neighbourhoods by price
print(nyc_cat_price_df[0:11]['neighbourhood'])

# Plot room type vs avg price of top 10 neighbourhoods
print("ROOM TYPE\n")
for neighbourhood in nyc_cat_price_df[0:10]['neighbourhood']:
  avg_room_type_df = nyc_data_clean.loc[nyc_data_clean['neighbourhood'] == neighbourhood].groupby('room_type', as_index=False).mean()
  if avg_room_type_df.shape[0] == 1:
    plt.figure(figsize = (2, 6))
  elif avg_room_type_df.shape[0] == 2:
    plt.figure(figsize = (4, 6))
  else:
    plt.figure(figsize = (6, 6))
  bar = sns.barplot(x='room_type', y='price', data=avg_room_type_df).set_title(neighbourhood)

# Plot room type vs frequency of top 10 neighbourhoods
print("FREQUENCY\n")
for neighbourhood in nyc_cat_price_df[0:10]['neighbourhood']:
  freq_room_type_df = nyc_data_clean.loc[nyc_data_clean['neighbourhood'] == neighbourhood].groupby('room_type', as_index=False).count()
  if freq_room_type_df.shape[0] == 1:
    plt.figure(figsize = (2, 6))
  elif freq_room_type_df.shape[0] == 2:
    plt.figure(figsize = (4, 6))
  else:
    plt.figure(figsize = (6, 6))
  bar = sns.barplot(x='room_type', y='price', data=freq_room_type_df).set_title(neighbourhood)