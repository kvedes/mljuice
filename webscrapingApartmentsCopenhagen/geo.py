import pandas as pd
import re
import requests
import utm

df = pd.read_csv('preGeo.csv')

#######
# Function: DAWA geocoder
# input:  PandasDataframe
# output: PandasDataframe with appended columns of coordinates
# Desc: It uses the adresses to obtain coordinates
# f(x.adress) = x.coordinate
# It uses the open DAWA API with no request limit
#######

def DAWA_geocode(df):

    df['housenumber'] =df.streetaddress.apply(lambda m: re.search('(\d+.?)', m).group(0))
    df.housenumber=df.housenumber.str.replace(r"[ \t]+$", '')

    df.streetaddress=df.streetaddress.apply(lambda m: re.search('\D+', m).group(0))
    df.streetaddress=df.streetaddress.str.replace(r"[ \t]+$", '')
    df.streetaddress=df.streetaddress.str.replace("Alle", 'Allé')
    df.streetaddress=df.streetaddress.str.replace("C. F.", 'C.F.')

    url = "https://dawa.aws.dk/adresser"   #?vejnavn=Lilledal&husnr=23&postnr=3450
    for house in df.index:
        print("\r%d%%" % int(house/df.shape[0]*100))
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

        location = results[0]['adgangsadresse']['adgangspunkt']['koordinater']
        df.loc[house,'Y'] = location[0]
        df.loc[house,'X'] = location[1]

    return(df)

#######
# Function : Google geocoder
# input:  PandasDataframe
# output: PandasDataframe with appended columns of coordinates
# Desc: It uses the adresses to obtain coordinates
# f(x.adress) = x.coordinate
# It uses the google api  which has a limit of 2500 requests
# more requests can be made using the google dev API_KEY
#######

def Google_geocode(df):
    for house in df.index:
        adress = df.iloc[house].streetaddress
        if adress[-1].isalpha():
            adress = adress[:-1]
        city = df.iloc[house].city
        params = {'sensor': 'false', 'address': adress+' '+city}
        r = requests.get(url, params=params)
        results = r.json()['results']
        location = results[0]['geometry']['location']
        df.loc[house,'X'] = location['lat']
        df.loc[house,'Y'] = location['lng']


#######
# Function : UTM converter
# input:  PandasDataframe
# output: PandasDataframe with appended columns of UTM-coordinates
# desc: Returns UTM coordinates with datum WSG84
# f(x.coordinate) = x.utm-coordinate
#######

def UTM_coder(df):

    df['UTM1'] = 0
    df['UTM2'] = 0
    j=0
    for j in range(df.shape[0]):
            df.loc[j,['UTM1','UTM2']]= list(utm.from_latlon(df.X[j],df.Y[j]))[0:2]

    return(df)



df = DAWA_geocode(df)
df = UTM_coder(df)

df.to_csv('housingdata.csv',index=False)
