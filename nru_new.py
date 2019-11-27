

# Import pandas and load data:
import pandas as pd
df = pd.read_csv('/media/khanhan/New Volume/FuntapData/nru_new.csv')

# Remove missing for appsflyer_id:
appflyer = "appsflyer_id"
df_full = df[df[appflyer].notnull()]

# Time execution"
import time
start = time.time()
df = pd.read_csv('/home/khanhan/Desktop/FuntapData/nru_new.csv')
time_execution = time.time() - start
print(time_execution)












