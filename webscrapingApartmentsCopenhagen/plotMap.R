setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(spdep)
library(plotGoogleMaps)
library(googleVis)
library(dplyr)

mp <- RColorBrewer::brewer.pal(11,"Spectral")[8:1]

df <- read.csv('data/housingdata.csv')
df[,c(1,2)] <- NULL

cCols <- c("UTM1","UTM2")
df[duplicated(df[,cCols]), cCols] <-  df[duplicated(df[,cCols]), cCols] +
  matrix(rnorm(sum(duplicated(df[,cCols]))*2),ncol = 2)*8
df <- df %>% filter(UTM1<4e5)

coords <- df %>% select(UTM1,UTM2)
df$hlinks <- sprintf("<a href='%s' >Info</a>",df$links)
sp = SpatialPointsDataFrame(coords, df, proj4string = CRS("+proj=utm +zone=33 +ellps=WGS84 +units=m +no_defs"))

ic =iconlabels(round(sp$Price/10^3,3),
               colPalette=mp,
               at=NULL,
               height=16,
               icon=F,
               scale=0.6)

m<-plotGoogleMaps(sp,
                  zcol="Price",
                  filename='plots/mymap.html',
                  mapTypeId = 'HYBRID',
                  iconMarker=ic,
                  colPalette=mp,
                  legend = F,
                  control = FALSE,
                  openMap=T)
