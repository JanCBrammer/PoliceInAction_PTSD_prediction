# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 07:46:28 2019

@author: Jan C. Brammer <jan.c.brammer@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt

'''
The motivation behind this demo is to assure that interpolating the heart-period
and breathing-period data at 10Hz doesn't result in significant changes in the
accuracy of event-timing (with a desired accuracy of about half a heart-period
at fast HR); it doesn't (event with 10 samples per second, the accuracy is of
by at most 100 msec).
'''

sec = 26
sfreq1 = 16
sfreq2 = 500
sfreq3 = 5000

# Create vectors from 0 to sec, containing samples that are spaced at
# intervals of 1/sfreq.
x1 = np.linspace(0, sec, sec * sfreq1)
x2 = np.linspace(0, sec, sec * sfreq2)
x3 = np.linspace(0, sec, sec * sfreq3)

# Plot an event occuring at 5.2 seconds.
eventx = 5.2
eventx1 = int(np.rint(eventx * sfreq1))
eventx2 = int(np.rint(eventx * sfreq2))
eventx3 = int(np.rint(eventx * sfreq3))

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax0.scatter(x1, np.ones(len(x1)))
ax0.axvline(x=x1[eventx1], c='r')
ax1.scatter(x2, np.ones(len(x2)))
ax1.axvline(x=x2[eventx2], c='r')
ax2.scatter(x3, np.ones(len(x3)))
ax2.axvline(x=x3[eventx3], c='r')


