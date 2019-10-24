import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

boston_data = pd.read_csv('data/listings.csv')

# =========== Data Visualization =========== 
# print("boston shape: ", boston_data.shape)
# print("boston column names: ", boston_data.columns)

boston_col = boston_data.columns
boston_num_rows = boston_data.shape[0]


# =========== Data Cleaning ===========

del_boston_cols = ["id", "listing_url", "scrape_id", "last_scraped", "name", "summary", "space", "description", "experiences_offered", "neighborhood_overview", "notes", "transit", "access", "interaction", "house_rules",
 "thumbnail_url", "medium_url", "picture_url", "xl_picture_url", "host_id", "host_url", "host_name", "host_since", "host_location", "host_about", "host_response_time", "host_response_rate", "host_acceptance_rate", 
 "host_is_superhost", "host_thumbnail_url", "host_picture_url", "host_neighbourhood", "host_listings_count", "host_total_listings_count", "host_verifications", "host_has_profile_pic", "host_identity_verified", "street", 
 "neighbourhood", "neighbourhood_group_cleansed", "state", "zipcode", "market", "smart_location", "country_code", "country", "is_location_exact", "property_type", "accommodates", "bathrooms", "bedrooms", "beds", "bed_type", 
 "amenities", "square_feet", "weekly_price", "monthly_price", "security_deposit", "cleaning_fee", "guests_included", "extra_people", "maximum_nights", "calendar_updated", "has_availability", "availability_30", "availability_60", 
 "availability_90", "calendar_last_scraped", "first_review", "last_review", "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", 
 "review_scores_value", "requires_license", "license", "jurisdiction_names", "instant_bookable", "cancellation_policy", "require_guest_profile_picture", "require_guest_phone_verification", "minimum_nights", 'city']

# From "price", remove $ and turn to int
boston_price_feature = boston_data['price'].apply(lambda x: x.split('.')[0]).replace('[^0-9]', '', regex=True).apply(lambda x: int(x)) 

# Put price in numpy array
boston_price_feature = boston_price_feature.to_numpy()
boston_data['price'] = boston_price_feature

# Remove columns
boston_data_clean = boston_data.drop(del_boston_cols, axis=1)
# boston_data_clean.head()

# Split continuous and categorical features
boston_cts_feats = ['latitude', 'longitude', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
boston_cat_feats = ['neighbourhood_cleansed', 'room_type']

boston_cts_data = boston_data_clean[boston_cts_feats].copy()
boston_cat_data = boston_data_clean[boston_cat_feats].copy()

# =========== Boston Continuous Data ===========
boston_price_feature_df = pd.DataFrame(boston_price_feature)

# Replace NaN in reviews_per_month with 0
boston_data_clean['reviews_per_month'] = boston_data_clean['reviews_per_month'].fillna(0)

# Get correlation of of price vs continous data
for feature in boston_cts_data:
  corr, _ = pearsonr(boston_price_feature, boston_data_clean[feature])
  print('Pearsons correlation of ' + feature + ': %.3f' % corr)
  
  # Plot correlation between price and feature
  concat_df = pd.concat([boston_price_feature_df, boston_data_clean[feature]], ignore_index=True, axis=1)
  concat_df.columns = ['price', feature]
  #sns.pairplot(concat_df)

  # Pair plot of every continous feature vs each other, too long to run
  # boston_price_feature_df = pd.concat([boston_price_feature_df, boston_data_clean[feature]], ignore_index=True, axis=1)

# Pair plot of every continous feature vs each other, too long to run
# boston_price_feature_df.columns = ['price'] + boston_cts_feats
# sns.pairplot(boston_price_feature_df)


# =========== Boston Categorical Data ===========

# Print the mean price by each categorical data, sorted from highest to lowest. 
# Store sorted lists of each categorical feature in boston_cat_price
boston_cat_price = []
for feature in boston_cat_feats:
  print("\n" + feature)
  list_feats = []
  for unique in boston_data_clean[feature].unique():
    list_feats.append([boston_data[boston_data[feature] == unique]['price'].mean(), unique])
  
  list_feats.sort(reverse=True)
  boston_cat_price.append(list_feats)

  for i in list_feats:
    print(i[1], "%.2f" % i[0])

# Create new dataframes of price and each categorical feature and plot
boston_cat_price_df = pd.DataFrame(boston_cat_price[0])
boston_cat_price_df.columns = ['price', 'neighbourhood_cleansed']
boston_cat_price_df2 = pd.DataFrame(boston_cat_price[1])
boston_cat_price_df2.columns = ['price', 'room type']

# Plot price vs each categorical data
for df in [boston_cat_price_df, boston_cat_price_df2]:
  plt.figure(figsize = (6, 6))
  bar = sns.barplot(x = df.columns[1], y = 'price', data=df)
  xt = plt.xticks(rotation=90)
  fig = bar.get_figure()

# Top 10 neighbourhoods by price
print(boston_cat_price_df[0:11]['neighbourhood_cleansed'])

# Plot room type vs avg price of top 10 neighbourhoods
print("ROOM TYPE\n")
for neighbourhood in boston_cat_price_df[0:10]['neighbourhood_cleansed']:
  avg_room_type_df = boston_data_clean.loc[boston_data_clean['neighbourhood_cleansed'] == neighbourhood].groupby('room_type', as_index=False).mean()
  if avg_room_type_df.shape[0] == 1:
    plt.figure(figsize = (2, 6))
  elif avg_room_type_df.shape[0] == 2:
    plt.figure(figsize = (4, 6))
  else:
    plt.figure(figsize = (6, 6))
  bar = sns.barplot(x='room_type', y='price', data=avg_room_type_df).set_title(neighbourhood)

# Plot room type vs frequency of top 10 neighbourhoods
print("FREQUENCY\n")
for neighbourhood in boston_cat_price_df[0:10]['neighbourhood_cleansed']:
  freq_room_type_df = boston_data_clean.loc[boston_data_clean['neighbourhood_cleansed'] == neighbourhood].groupby('room_type', as_index=False).count()
  if freq_room_type_df.shape[0] == 1:
    plt.figure(figsize = (2, 6))
  elif freq_room_type_df.shape[0] == 2:
    plt.figure(figsize = (4, 6))
  else:
    plt.figure(figsize = (6, 6))
  bar = sns.barplot(x='room_type', y='price', data=freq_room_type_df).set_title(neighbourhood)