import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import mpl_settings  # Fancy formatting rules for plots.

fhgt = 10  # Height of figures in inch.
fwid = 16  # Width of figures in inch.
mpl_settings.set_plot_params(useTex=True)  # Make plots look good.
plt.ion()
rows, columns = os.popen('stty size', 'r').read().split()
np.set_printoptions(linewidth=int(columns) - 2)

Overnight = 8.0/24.0  # Simplistic duration of day that we want to consider as a single night rental.

# Read in the data as a pandas DataFrame.
dataFile = pd.read_excel('./data/TT_Dataset_for_Data_Scientist_Candidate.xlsx',
                         sheetname=1,
                         header=0)
numRecords = dataFile['Account ID'].count()
lastRec = numRecords - 1

# Compute and add the duration of rental times to the dataFile.
maxRentalPossible = (pd.Timestamp(dataFile['ReserveDate'].values[numRecords - 1]) -
                     pd.Timestamp(dataFile['ReserveDate'].values[0]))/np.timedelta64(1, 'D')

notRented = np.ones(numRecords, dtype=bool)
badReserveDate = np.zeros(numRecords, dtype=bool)
badMoveInDate = np.zeros(numRecords, dtype=bool)
badMoveOutDate = np.zeros(numRecords, dtype=bool)
badDate = np.zeros(numRecords, dtype=bool)
durationArr = np.zeros(numRecords, dtype=float)
notMovedOutArr = np.zeros(numRecords, dtype=bool)
infoStr = 'ID: %d; Rented? %r; ReserveDate: %s; MoveInDate: %s; MoveOutDate %s; Duration (hrs): %f; Promo: %s'
for recNum in xrange(numRecords):
    # Convert all dates to pd.Timestamp for convenience
    dataFile['ReserveDate'].values[recNum] = pd.Timestamp(dataFile['ReserveDate'].values[recNum])
    dataFile['Move In Date'].values[recNum] = pd.Timestamp(dataFile['Move In Date'].values[recNum])
    if not pd.isnull(dataFile['Move Out Date'].values[recNum]):
        dataFile['Move Out Date'].values[recNum] = pd.Timestamp(
            dataFile['Move Out Date'].values[recNum])

    # Check for bad reserve date.
    if ((dataFile['ReserveDate'].values[recNum] < dataFile['ReserveDate'].values[0]) or
            (dataFile['ReserveDate'].values[recNum] > dataFile['Move In Date'].values[lastRec])):
        badReserveDate[recNum] = True
    # Check for bad move in date
    if ((dataFile['Move In Date'].values[recNum] < dataFile['ReserveDate'].values[0]) or
            (dataFile['Move In Date'].values[recNum] > dataFile['Move In Date'].values[lastRec])):
        badMoveInDate[recNum] = True
    # If rented, check if moved out and compute duration of stay.
    if dataFile['Rented?'].values[recNum]:  # Was the unit rented?
        if not pd.isnull(dataFile['Move Out Date'].values[recNum]):  # Do we know the move out date?
            # Check for bad move out date.
            if ((dataFile['Move Out Date'].values[recNum] < dataFile['ReserveDate'].values[0]) or
                    (dataFile['Move Out Date'].values[recNum] > dataFile['Move In Date'].values[lastRec]) or
                    (dataFile['Move Out Date'].values[recNum] < dataFile['Move In Date'].values[recNum])):
                badMoveOutDate[recNum] = True
            # Compute rental duration = move out date - move in date
            duration = dataFile['Move Out Date'].values[recNum] - dataFile['Move In Date'].values[recNum]
            notMovedOut = True
        else:
            # Rental still occupied! Compute rental duration = last reserve date - move in date
            duration = dataFile['Move In Date'].values[lastRec] - dataFile['Move In Date'].values[recNum]
            notMovedOut = False
        durationArr[recNum] = duration/np.timedelta64(1, 'D')
        # Really we should check the actual timestamps but this is much simpler.
        if durationArr[recNum] < Overnight:
            print infoStr%(dataFile['Account ID'].values[recNum],
                           dataFile['Rented?'].values[recNum],
                           str(dataFile['ReserveDate'].values[recNum]),
                           str(dataFile['Move In Date'].values[recNum]),
                           str(dataFile['Move Out Date'].values[recNum]),
                           durationArr[recNum]*24,
                           dataFile['Promotion Name'].values[recNum])
        notMovedOutArr[recNum] = notMovedOut
        badDate[recNum] = badReserveDate[recNum] + badMoveInDate[recNum] + badMoveOutDate[recNum]
        notRented[recNum] = False
    else:
        notRented[recNum] = True
dataFile['Not Rented?'] = notRented
dataFile['Bad Reserve Date'] = badReserveDate
dataFile['Bad Move In Date'] = badMoveInDate
dataFile['Bad Move Out Date'] = badMoveOutDate
dataFile['Bad Date'] = badDate
dataFile['Rental Duration'] = durationArr
dataFile['Not Moved Out?'] = notMovedOutArr

binList = [0.0, 0.1, Overnight, 1.0, 30.0, 60.0, 90.0, 120, 150, 180.0, 210, 240, 270.0, 300.0, 330.0, 360.0,
           390.0, 420.0, 450.0, 480.0, 510.0, 540.0, 570.0, 600.0]
goodData = dataFile['Rental Duration'].values[dataFile['Not Rented?'].values +
                                              dataFile['Bad Date'].values +
                                              dataFile['Not Moved Out?'].values == 0]
histOut = plt.hist(
    goodData,
    bins=binList, normed=False, facecolor='green', alpha=0.75)

pdb.set_trace()
