# webscraping

Code for the blog post [housing data](http://www.mljuice.com/webscraping-apartments-copenhagen) regarding webscraping of housing data in copenhagen.

scraping.py contains the code for scraping data from [findbolig](http://www.findbolig.nu).

wrangling.py takes care of tidying up the scraped data using regex and other tricks

geo.py shows how to geocode the addresses using either [DAWA](https://dawa.aws.dk/) or [Google Maps](https://developers.google.com/maps/documentation/geocoding/start).

Everything can be run in conjunction using the master.py, it simply calls scraping.py,wrangling.py and geo.py sequentially.

The multithreadingtest.py shows some statistics regarding the speedup gained from using the multiprocess or threading library in python as opposed to the standard seq.
