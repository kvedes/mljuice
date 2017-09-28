from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import codecs
import numpy as np
import time
import tqdm

#######
# Function
# input:  page number, default =1
# output: the body of the html page
# Desc: It simply returns all tags in between <body></body>
#######


def htmlbody(i=0):
    url = ('https://www.findbolig.nu/ledigeboliger/liste.aspx?&showrented=1&showyouth'+
    '=0&showlimitedperiod=1&showunlimitedperiod=1&showOpenDay=0&sortkey=AvailableDate&sortdir=asc&page='+str(i+1)+
    '&pagesize=100')
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page.read(),"html5lib")
    body = soup.find_all("tbody")
    return(body)

#######
# Clumsy way of finding number of pages.
# finds total number of listed apartments, devides by apartments per pages
# and uses ceil to give number of pages
#######

for element in htmlbody():
    xl = element.find_all("span")
    for idx,item in enumerate(xl):
        if('id' in item.attrs):
            if(item.attrs['id']=='ctl00_placeholdercontent_0_FbnTable1_lab_ResidencesFoundTop'):
                s=item.get_text()[0:4]
                s=s.replace(' ','')
numberofpages = int(np.ceil(pd.to_numeric(s)/100))

#######
# Loops over every page with listed apartments, taking an element into
# a list when an "if" statement is satisfied
#######
from threading import Thread,Lock

bodies = []
lock = Lock()
def threadSafelist(i):
    threadRes=htmlbody(i)
    lock.acquire() # will block if lock is already held
    bodies.append(threadRes)
    lock.release()

threads = []
for i in tqdm.tqdm(range(numberofpages)):#searchTerms:
    threads.append (Thread (target=threadSafelist, args=(i,)))
    threads[-1].start()

for t in threads:
    t.join()

#######
# Initializing lists for the wanted information
#######

link_list   =  []
adress_list =  []
size_list   =  []
room_list   =  []
price_list   = []
avail_list   = []
Owner_list   = []


for body in bodies:

    for element in body:
        check=0
        xl = element.find_all("img")

        #######
        # Example of if statement:
        # Extracting owning firmname, by taking all tags <img>
        # and checking if owner is a listed attribute
        # However 1 firm is not listed this way and must be found in the src
        # attribute, so it can be a little bit tricky and some fiddling is
        # necessary
        #######

        for idx,item in enumerate(xl):
            if ('title' in item.attrs):
                Owner_list.append(item.attrs['title'])
            if (('src' in item.attrs) & ('title' not  in item.attrs)):
                temp = item.attrs['src']
                if(temp[1:4]=='FBN'):
                    if(check==1):
                        Owner_list.append('AdvokaterLoegstrup')
                    check = 1
            else:
                check = 0

        xl = element.find_all("a", class_='advertLink')
        for idx,item in enumerate(xl):
            if (item.get_text()!=''):
                adress_list.append(item.get_text())
            if('href' in item.attrs):
                link_list.append(item.attrs['href'])

        xl = element.find_all("td")
        for idx,item in enumerate(xl):

            if list(item.attrs.items()).__len__()>0:
                if (list(item.attrs.values())[0]=='text-align:center;'):
                    if int(item.get_text()) < 10:
                        room_list.append(item.get_text())
                    else:
                        size_list.append(item.get_text())

                if list(item.attrs.values())[0]=='text-align:right;':
                    if (('Ledig' in item.get_text())|('-' in item.get_text())):
                        avail_list.append(item.get_text())
                    else:
                        price_list.append(item.get_text())


df = pd.DataFrame({'Price':price_list,'Size': size_list, 'Rooms': room_list,
                   'Address':adress_list, 'Availability':avail_list, 'Owner': Owner_list,
                   'links': list(pd.unique(link_list))})

df.to_csv("data/preWrangling.csv", index=False)
