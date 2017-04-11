from __future__ import print_function,division
import numpy as np
import DarkChannelRecover
import dehaze
a=np.arange(start=0,stop=300)
a=np.reshape(a,newshape=(10,10,3))
#print (a)


#get minchannel test
a_min=DarkChannelRecover.getMinChannel(a)
#print (a_min)

b_min=dehaze.getMinMap(a)
#print (b_min)

#get dark img
a_dark=DarkChannelRecover.getDarkChannel(a_min)
#print(a_dark)
#print ("\n")
pad,b_dark=dehaze.getDarkMap(b_min)
#print (b_dark)

#get atmos
a_atmos=DarkChannelRecover.getAtomsphericLight(a_dark,a,meanMode=True)
print (a_atmos)

b_atmos=dehaze.getAtmosphericLight(b_dark)
print (b_atmos)