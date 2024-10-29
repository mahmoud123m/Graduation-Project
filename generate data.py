import pandas as pd
import numpy as np

# Define the number of cities and users
num_place = 39
num_users = 1000
# Generate random data
rng = np.random.default_rng()
data = {
    # 'userID': rng.choice(num_users, size= (num_users), replace=False),
    'placeID': np.random.randint(1, num_place , size= num_users),
    # 'place_rating': np.random.uniform(1, 5, size= num_users),
    'hotelID': np.random.randint(1, 20, size= num_users),
    # 'hotel_rating': np.random.uniform(1, 5, size= num_users),
    'restaurantID': np.random.randint(1, 18, size= num_users),
    # 'restaurant_rating': np.random.uniform(1, 5, size= num_users),
    'eventID': np.random.randint(1, 19, size= num_users),
    # 'event_rating': np.random.uniform(1, 5, size= num_users)
}

# Create a DataFrame
print(data.items())

df = pd.DataFrame(data)
# df=df.sort_values(by=['userID'])
df['placeID'] = '0' + df['placeID'].astype(str)
df['hotelID'] = '1' + df['hotelID'].astype(str)
df['restaurantID'] = '2' + df['restaurantID'].astype(str)
df['eventID'] = '3' + df['eventID'].astype(str)
print(df)
# df= df.iloc[20:]
# print(df)
# Save the data to a CSV file
df.to_csv('data.csv', index=False)
