# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:57:01 2019

@author: John Doe
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

path = r'QuestionD_AllParticipSelect.xlsx'

df = pd.read_excel(path, sheet_name='POL_coreTraumaW2W1',
                   usecols=['PCL_w1_Sum', 'PCL_w2_Sum', 'PCL_w2MINw1_sum',
                            'PLES_w1_Sum', 'PLES_w2_Sum',
                            'PSS_w1_Sum', 'PSS_w2_Sum',
                            'BDI_w1_Sum', 'BDI_w2_Sum'])
df.dropna(inplace=True)

fig = plt.figure()
gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:, 1])

# center bins on x-ticks (-0.5)
sb.distplot(df['PCL_w1_Sum'],
            bins=np.arange(int(df['PCL_w1_Sum'].min()),
                           int(df['PCL_w1_Sum'].max()+1)) - 0.5,
                           ax=ax1)
sb.distplot(df['PCL_w2_Sum'],
            bins=np.arange(int(df['PCL_w2_Sum'].min()),
                           int(df['PCL_w2_Sum'].max()+1)) - 0.5,
                           ax=ax2)
sb.distplot(df['PCL_w2MINw1_sum'],
            bins=np.arange(int(df['PCL_w2MINw1_sum'].min()),
                           int(df['PCL_w2MINw1_sum'].max()+1)) - 0.5,
                           ax=ax3)
ax3.axvline(x=0, c='r')

increase = 0
decrease = 0
for i in df.index.tolist():
    pre = df.loc[i, 'PCL_w1_Sum']
    post = df.loc[i, 'PCL_w2_Sum']
    if pre > post:
        decrease += 1
        color = 'b'
    elif pre < post:
        increase += 1
        color = 'y'
    ax4.scatter([0, 1], [pre, post], c='b')
    ax4.plot([0, 1], [pre, post], color, alpha=0.5)
print(increase, decrease)

fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

# center bins on x-ticks (-0.5)
sb.distplot(df['PLES_w1_Sum'],
            bins=np.arange(int(df['PLES_w1_Sum'].min()),
                           int(df['PLES_w1_Sum'].max()+1)) - 0.5,
                           ax=ax1)
sb.distplot(df['PLES_w2_Sum'],
            bins=np.arange(int(df['PLES_w2_Sum'].min()),
                           int(df['PLES_w2_Sum'].max()+1)) - 0.5,
                           ax=ax2)

increase = 0
decrease = 0
for i in df.index.tolist():
    pre = df.loc[i, 'PLES_w1_Sum']
    post = df.loc[i, 'PLES_w2_Sum']
    if pre > post:
        decrease += 1
        color = 'b'
    elif pre < post:
        increase += 1
        color = 'y'
    ax3.scatter([0, 1], [pre, post], c='b')
    ax3.plot([0, 1], [pre, post], color, alpha=0.5)
print(increase, decrease)


fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

# center bins on x-ticks (-0.5)
sb.distplot(df['BDI_w1_Sum'],
            bins=np.arange(int(df['BDI_w1_Sum'].min()),
                           int(df['BDI_w1_Sum'].max()+1)) - 0.5,
                           ax=ax1)
sb.distplot(df['BDI_w2_Sum'],
            bins=np.arange(int(df['BDI_w2_Sum'].min()),
                           int(df['BDI_w2_Sum'].max()+1)) - 0.5,
                           ax=ax2)

increase = 0
decrease = 0
for i in df.index.tolist():
    pre = df.loc[i, 'BDI_w1_Sum']
    post = df.loc[i, 'BDI_w2_Sum']
    if pre > post:
        decrease += 1
        color = 'b'
    elif pre < post:
        increase += 1
        color = 'y'
    ax3.scatter([0, 1], [pre, post], c='b')
    ax3.plot([0, 1], [pre, post], color, alpha=0.5)
print(increase, decrease)

