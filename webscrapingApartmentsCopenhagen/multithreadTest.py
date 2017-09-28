import pandas as pd
import re
import requests
import utm
import tqdm
import time
import os

os.chdir("/home/christian/Documents/python/boligmodellering/blog/")

df = pd.read_csv('preGeo.csv')

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

def dawa_Logic_i(house):
    road = df.iloc[house].streetaddress
    number = df.iloc[house].housenumber
    postal = df.iloc[house].postal
    params = {'vejnavn': re.sub('é','e',road), 'husnr': number.upper(), 'postnr': postal}
    r = requests.get(url, params=params)
    results = r.json()
    if results ==[]:
        params = {'vejnavn': road, 'husnr': 1, 'postnr': postal}
        r = requests.get(url, params=params)
        results = r.json()
    return(results[0]['adgangsadresse']['adgangspunkt']['koordinater']+[house])



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

def DAWA_geocode(df):

    res = []
    for house in tqdm.tqdm(df.itertuples(), total=df.shape[0]):
        res.append(dawa_Logic(house))

    return(res)



print("Benchmark sequential execution:")
start_time = time.time()
listofkoordinates = DAWA_geocode(df)
print("Benchmark sequential execution done in %0.2f minutes." % ((time.time() - start_time)/60)  )

print("Multiprocessing using Pool:")
start_time = time.time()
from multiprocessing import Pool,cpu_count

p = Pool(cpu_count()-1)
multiple_results = [p.apply_async(dawa_Logic_i, (house,)) for house in df.index]


res2 = [res.get(timeout=10) for res in tqdm.tqdm(multiple_results)]

res2
print("multiprocessing using Pool done in %0.2f minutes." % ((time.time() - start_time)/60)  )



print("Multiprocessing using threading:")
start_time = time.time()
from threading import Thread,Lock

res1 = []
lock = Lock()
def threadSafelist(house):
    threadRes=dawa_Logic(house)
    lock.acquire() # will block if lock is already held
    res1.append(threadRes)
    lock.release()

threads = []
for house in tqdm.tqdm(df.itertuples(),total=df.shape[0]):#searchTerms:
    threads.append (Thread (target=threadSafelist, args=(house,)))
    threads[-1].start()

for t in threads:
    t.join()

res1
print("multiprocessing using threading done in %0.2f minutes." % ((time.time() - start_time)/60)  )

# from pandas.util.testing import assert_frame_equal
#
# assert_frame_equal(pd.DataFrame(res, columns=['X','Y','Index']).sort("Index").set_index(['Index']),
#                     pd.DataFrame(res1, columns=['X','Y','Index']).sort("Index").set_index(['Index']) )
#
# assert_frame_equal(pd.DataFrame(res1, columns=['X','Y','Index']).sort("Index").set_index(['Index']),
#                     pd.DataFrame(res2, columns=['X','Y','Index']).sort("Index").set_index(['Index']) )
#

df = pd.merge(df, pd.DataFrame(res1, columns=['X','Y','Index']).set_index(['Index']), left_index=True, right_index=True, how='outer')

df['UTM1'] = 0
df['UTM2'] = 0
df[['UTM1','UTM2']] = df.apply(lambda x: list(utm.from_latlon(x['X'], x['Y'])[0:2]), axis=1).apply(pd.Series)

df.to_csv('housingdata.csv',index=False)
