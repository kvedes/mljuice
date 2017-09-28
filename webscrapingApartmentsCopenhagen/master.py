import time

print("Starting to scrape:")
start_time = time.time()
import scraping
print("Scraping done in %0.2f minutes." % ((time.time() - start_time)/60)  )

print("")
print("Starting to data wrangle:")
start_time = time.time()
import scraping
print("Wrangling done in %0.2f minutes." % ((time.time() - start_time)/60)  )

import wrangling

print("")
print("Starting to Geocode:")
start_time = time.time()
import geo
print("Geocoding done in %0.2f minutes." % ((time.time() - start_time)/60)  )



import pandas as pd

geo.df
