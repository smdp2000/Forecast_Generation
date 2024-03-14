import requests
import pandas as pd
import numpy as np
from datetime import datetime

URL  = 'https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD'

response = requests.get(sel_url)
with open('dummy.csv', 'wb') as f:
    f.write(response.content)
hosp_tab = pd.read_csv('dummy.csv')
popu = np.loadtxt('us_states_population_data.txt') #population of each state
abvs = pd.read_csv('us_states_abbr_list.txt', header=None)[0].tolist()
ns = len(abvs) #no of states


data = []
with open('reich_fips.txt', 'r') as file:
    for line_num, line in enumerate(file, 1):
        fields = line.split(',')
        #print(len(fields))
        #print(fields)
        if len(fields) == 4:
            data.append(fields)
        else:
            print(f"Skipping line {line_num}: {line.strip()}")

df = pd.DataFrame(data, columns=['col1', 'col2', 'col3', 'col4'])
df['col4'] = df['col4'].astype(str)
df['col4'] = df['col4'].str.replace('\n', '')
df.columns = df.iloc[0]
df = df.drop(df.index[0]).reset_index(drop=True)
df['population'] = pd.to_numeric(df['population'])
fips_tab = df

filtered_fips_tab = fips_tab[~fips_tab['abbreviation'].isin([None, '', np.nan])]

# Create the dictionary
location_to_abbreviation = pd.Series(filtered_fips_tab.abbreviation.values, index=filtered_fips_tab.location).to_dict()

print(location_to_abbreviation)
location_mapper = location_to_abbreviation

hosp_tab['date'] = pd.to_datetime(hosp_tab['date'], format='%Y/%m/%d')
all_days = (hosp_tab['date'] - zero_date).dt.days
bad_idx = all_days <= 0
hosp_tab = hosp_tab[~bad_idx]
all_days = all_days[~bad_idx]
maxt = all_days.max() - days_back

fips = [None] * ns
hosp_dat = pd.DataFrame(index=range(ns), columns=range(maxt))
for cid in range(ns):
    fips[cid] = fips_tab[fips_tab['abbreviation'] == abvs[cid]]['location'].values[0]

for idx in range(len(hosp_tab)):
    state = hosp_tab.iloc[idx]['state']
    cid = abvs.index(state) if state in abvs else None
    if cid is None:
        print(f'Error at {idx}')
    else:
        date_idx = all_days.iloc[idx]
        if date_idx-1 <maxt:
            hosp_dat.at[cid, date_idx-1] = hosp_tab.iloc[idx]['previous_day_admission_influenza_confirmed']

hosp_cov = np.full((ns, maxt), np.nan)
for idx in range(len(hosp_tab)):
    #print('idx', idx)
    state = hosp_tab.iloc[idx]['state']
    cid = abvs.index(state) if state in abvs else None
    if cid is not None:
        date_idx = all_days.iloc[idx]
        if date_idx-1 <maxt:
            print(cid,idx)
            hosp_dat.at[cid, date_idx-1] = hosp_tab.iloc[idx]['previous_day_admission_influenza_confirmed']
            hosp_cov[cid, date_idx-1] = hosp_tab.iloc[idx]['previous_day_admission_influenza_confirmed_coverage']


np.savetxt('hosp_cov.csv', hosp_cov, delimiter=',')
hosp_dat.to_csv('hosp_dat.csv', index=False)

