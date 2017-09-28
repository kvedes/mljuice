import pandas as pd
import re
import requests
import utm
import tqdm
import time
import os

os.chdir("/home/christian/Documents/python/boligmodellering/blog/")

df = pd.read_csv('data/preGeo.csv')

#######
# Function : Google geocoder
# input:  PandasDataframe
# output: PandasDataframe with appended columns of coordinates
# Desc: It uses the adresses to obtain coordinates
# f(x.adress) = x.coordinate
# It uses the google api  which has a limit of 2500 requests
# more requests can be made using the google dev API_KEY
#######
def Google_geocode(house):
    road = house.streetaddress
    number = house.housenumber
    postal = house.city
    if number[-1].isalpha():
        number = number[:-1]
    params = {'sensor': 'false', 'address': road + number.upper() + ' '+city}
    r = requests.get('https://maps.googleapis.com/maps/api/geocode/json',
                    params=params)
    results = r.json()['results']
    return(results[0]['geometry']['location'])

#######
# Function: DAWA geocoder
# input:  PandasDataframe
# output: PandasDataframe with appended columns of coordinates
# Desc: It uses the adresses to obtain coordinates
# f(x.adress) = x.coordinate
# It uses the open DAWA API with no request limit
# Syntax for DAWA: #?vejnavn=Lilledal&husnr=23&postnr=3450
#######
url = "https://dawa.aws.dk/adresser"

def dawa_Logic(house):
    road = house.streetaddress
    number = house.housenumber
    postal = house.postal
    params = {'vejnavn': re.sub('é','e',road), 'husnr': number.upper(), 'postnr': postal}
    r = requests.get(url, params=params)
    results = r.json()
    if results ==[]:
        params = {'vejnavn': road, 'husnr': 1, 'postnr': postal}
        r = requests.get(url, params=params)
        results = r.json()
    return(results[0]['adgangsadresse']['adgangspunkt']['koordinater']+[house.Index])


#######
# As the most time consuming part of webscraping is the requests
# we will use multithreading to start a bunch of requests simoultaneously
#######
from threading import Thread,Lock

res = []
lock = Lock()
def threadSafelist(house):
    threadRes=dawa_Logic(house)
    lock.acquire() # only one thread writes to list at a time
    res.append(threadRes)
    lock.release()

threads = []
for house in tqdm.tqdm(df.itertuples(),total=df.shape[0]):#searchTerms:
    threads.append (Thread (target=threadSafelist, args=(house,)))
    threads[-1].start()

for t in threads:
    t.join()

#######
# Merging coordinates with dataframe based on index, as multithreading is
# bad at respecting order in a shared data-structure.
#######

df = pd.merge(df, pd.DataFrame(res, columns=['X','Y','Index']).set_index(['Index']), left_index=True, right_index=True, how='outer')

#######
# Returning UTM coordinates with datum WSG84
# f(x.coordinate) = x.utm-coordinate
#######

df['UTM1'] = 0
df['UTM2'] = 0
df[['UTM1','UTM2']] = df.apply(lambda x: list(utm.from_latlon(x['Y'], x['X'])[0:2]), axis=1).apply(pd.Series)

df.to_csv('data/housingdata.csv',index=False)
