# Erik Idéfil

## Transformering av data:
Visibility borde inte göra så stor skillnad? sqrt

### Behålla
Weekday
Holiday
Windspeed
Summertime

### Modifera:
hour_of_day, gör om till tre klasser
Month, sin och cos


### Göra om till booleans:
snowdepth
precipitation

### Ta bort:
Cloudcover
day_of_week
Snow, aldrig ens med

### Kvar:
Temp
Dew
Humidity
Visibility

nr   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   hour_of_day     1600 non-null   int64  
 1   day_of_week     1600 non-null   int64  
 2   month           1600 non-null   int64  
 3   holiday         1600 non-null   int64  
 4   weekday         1600 non-null   int64  
 5   summertime      1600 non-null   int64  
 6   temp            1600 non-null   float64
 7   dew             1600 non-null   float64
 8   humidity        1600 non-null   float64
 9   precip          1600 non-null   float64
 10  snow            1600 non-null   int64  
 11  snowdepth       1600 non-null   float64
 12  windspeed       1600 non-null   float64
 13  cloudcover      1600 non-null   float64
 14  visibility      1600 non-null   float64
 15  increase_stock  1600 non-null   object 