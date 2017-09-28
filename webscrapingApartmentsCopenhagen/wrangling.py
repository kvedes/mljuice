import pandas as pd

df = pd.read_csv('data/preWrangling.csv')

#######
# Appending site name to make valid urls from scraped links
#######

df['links'] = r'https://www.findbolig.nu' + df['links']
df=df.dropna(axis=0, how='any')

#######
# Using regular expression to extract concepts such as zip code,
# road name and city
#######

import re
df['postal']        = df.Address.apply(lambda m: re.search('\d{4}\s', m).group(0))
df['city']          = df.Address.apply(lambda m: re.search('(?<=\d{4}\s).*', m).group(0))#\w+
df['streetaddress']   = df.Address.apply(lambda m: re.split(',', m)[0])#\w+
df.streetaddress  = df.streetaddress.str.replace('  ', ' ')
df.streetaddress  = df.streetaddress.str.replace('  ', ' ')

#######
# Always filter as early as possible
# Here we remove houses not from Zealand
#######

df = df[(pd.to_numeric(df.postal.values)>=1000)&(pd.to_numeric(df.postal.values)<=4773)]
df=df.reset_index(drop=True)

#######
# Removing currency indicator and swapping decimal marker from DK -> US/EN
#######

df.Price = df.Price.str.replace('kr', '')
df.Price = df.Price.str.replace('.', '')
df.Price = df.Price.str.replace(',', '.')

#######
# Splitting aconto and price into 2 colummns
#######

pr_and_acconto = df.Price.str.extract('(?P<Price>\d*\.?\d*)\s(?P<acconto>\d*\.?\d*)', expand=False)
df = pd.concat([df.drop("Price",axis=1), pr_and_acconto], axis=1)

#######
# Convert strings to numeric for ease of use
#######

numeric_cols = ['Price','Rooms','Size','acconto','postal']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)


#######
# Using regular expression to extract concepts such as house number
# and street adress alone
#######

#df.streetaddress.str.extract('(\D*)\s(\d*.?)', expand=False)

df['housenumber'] =df.streetaddress.apply(lambda m: re.search('(\d+.?)', m).group(0))
df.housenumber=df.housenumber.str.replace(r"[ \t]+$", '')

df.streetaddress=df.streetaddress.apply(lambda m: re.search('\D+', m).group(0))
df.streetaddress=df.streetaddress.str.replace(r"[ \t]+$", '')
df.streetaddress=df.streetaddress.str.replace("Alle", 'Allé')
df.streetaddress=df.streetaddress.str.replace("C. F.", 'C.F.')


df.to_csv("data/preGeo.csv", index=False)
