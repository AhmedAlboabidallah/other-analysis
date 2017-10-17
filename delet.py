# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:00:33 2017

@author: ahalboabidallah
"""
import pandas as pd
table1=tables[0]
[field_data,RStable]=table1
RStable=sorted(RStable,key=lambda l:-float(l[7]))
Lowest_resolution=float(RStable[0][7])
fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error(field_data)# adds errors to the field data 
rss,Rsp_errorsX,Rsp_errorsY,Rsp_errorsT=add_error(RStable)# adds errors to the rss data 
Biomass,rs_bands=creat_table4corlation(rss,fields,RStable)# combines field data #combines any similar band and dataset RS inputs because they are subsetted from the same origional raster
Biomass1 = np.asarray(Biomass, dtype=float)
while len(Biomass1[0])==1:
     Biomass1=list(map(lambda x:x[0], Biomass1))
     print('<<>>')
Biomass1=list(filter(lambda a: a[2] > 0 and a[2] < 100000,Biomass1))
rs_bands[0]=list(filter(lambda a: a[2] > 0 and a[2] < 100000,rs_bands[0]))
yy=regression_images(rs_bands[0],Lowest_resolution,Lowest_resolution,Biomass1)
xx=[np.array(rs_bands[0])[:,2].tolist()]
for i in rs_bands[1:]:
        try:
            xx.append(regression_images(rs_bands[0],Lowest_resolution,Lowest_resolution,i))#
        except:
            print ('error')
   #for rs_data in rss:
    #        X=regression_images(list1,Lowest_resolution,Lowest_resolution,list(j))
    #        for i in range(1):
    #            print(i+1)
    #final_table.append(list(map(lambda x:x[0]**i+1, X)))#int(s) if s.isdigit() else 0
xx.append(yy)
    #xx=np.transpose(np.array(xx)).tolist
    #xx=pd.DataFrame(np.transpose(np.array(xx)))
xx=pd.DataFrame(xx)
xx=xx.transpose()
xx=xx.dropna()
xx=xx.values.tolist()
xx=list(map(lambda x:x[0:-2], xx))
yy=list(map(lambda x:x[-1], xx))
print('xx',xx)
print('yy',yy)
model = sm.api.OLS(yy,xx)#len(final_table[1:].values.tolist())==len(final_table[0].values.tolist())
results = model.fit()