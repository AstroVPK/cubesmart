import math
import os
import argparse
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import mpl_settings  # Fancy formatting rules for plots.

fhgt = 10  # Height of figures in inch.
fwid = 16  # Width of figures in inch.
DPI = 300  # Dots per inch for plots
mpl_settings.set_plot_params(useTex=True)  # Make plots look good.
rows, columns = os.popen('stty size', 'r').read().split()
np.set_printoptions(linewidth=int(columns) - 2)

Overnight = 8.0/24.0  # Simplistic duration of day that we want to consider as a single night rental.

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type=str, default=os.environ['PWD'], help=r'Path to working directory')
parser.add_argument('--plot', dest='plot', action='store_true', help=r'Make & show plot?')
parser.add_argument('--no-plot', dest='plot', action='store_false', help=r'Do not make & show plot?')
parser.set_defaults(plot=True)
parser.add_argument('--stop', dest='stop', action='store_true', help=r'Stop at end?')
parser.add_argument('--no-stop', dest='stop', action='store_false', help=r'Do not stop at end?')
parser.set_defaults(stop=True)
parser.add_argument('--save', dest='save', action='store_true', help=r'Save files?')
parser.add_argument('--no-save', dest='save', action='store_false', help=r'Do not save files?')
parser.set_defaults(save=True)
parser.add_argument('--pdf', dest='pdf', action='store_true', help=r'Save PDF figures?')
parser.add_argument('--no-pdf', dest='pdf', action='store_false', help=r'Do not save PDF figures?')
parser.set_defaults(pdf=False)
parser.add_argument('--rerun', dest='rerun', action='store_true', help=r'Re-run fit?')
parser.add_argument('--no-rerun', dest='rerun', action='store_false', help=r'Do not re-run fit?')
parser.set_defaults(rerun=False)
parser.add_argument('--ion', dest='ion', action='store_true', help=r'Display plots interactively?')
parser.add_argument('--no-ion', dest='ion', action='store_false', help=r'Do not display plots interactively?')
parser.set_defaults(ion=False)
parser.add_argument('--ratios', dest='ratios', action='store_true', help=r'Compute simple ratios?')
parser.add_argument('--no-ratios', dest='ratios', action='store_false', help=r'Do not compute simple ratios?')
parser.set_defaults(ratios=True)
args = parser.parse_args()

if args.ion:
    plt.ion()

# Read in the data as a pandas DataFrame
# ------------------------------------------------------------------------------------------------------------

if os.path.isfile(os.path.join(os.path.join(args.pwd, 'data/rentalData.pkl'))):
    print 'Reading in pickled data file...'
    dataFile = pickle.load(open(os.path.join(args.pwd, 'data/rentalData.pkl'), 'rb'))
    numRecords = dataFile['Account ID'].count()
    lastRec = numRecords - 1
else:
    print 'Reading in original excel file...'
    dataFile = pd.read_excel(os.path.join(args.pwd, 'data/TT_Dataset_for_Data_Scientist_Candidate.xlsx'),
                             sheetname=1,
                             header=0)
    numRecords = dataFile['Account ID'].count()
    lastRec = numRecords - 1

    # Now clean the data to remove bad entries etc...
    # --------------------------------------------------------------------------------------------------------

    notRented = np.ones(numRecords, dtype=bool)
    badReserveDate = np.zeros(numRecords, dtype=bool)
    badMoveInDate = np.zeros(numRecords, dtype=bool)
    badMoveOutDate = np.zeros(numRecords, dtype=bool)
    badDate = np.zeros(numRecords, dtype=bool)
    durationArr = np.zeros(numRecords, dtype=float)
    notMovedOutArr = np.zeros(numRecords, dtype=bool)
    infoStr = 'ID: %d; Rented? %r; ReserveDate: %s; MoveInDate: %s; MoveOutDate %s;\
               Duration (hrs): %f; Promo: %s'
    for recNum in xrange(numRecords):
        # Convert nan to No Promotion
        if pd.isnull(dataFile['Promotion Name'].values[recNum]):
            dataFile['Promotion Name'].values[recNum] = u'No Promotion'
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
                if ((dataFile['Move Out Date'].values[recNum] <
                     dataFile['ReserveDate'].values[0]) or
                    (dataFile['Move Out Date'].values[recNum] >
                     dataFile['Move In Date'].values[lastRec]) or
                    (dataFile['Move Out Date'].values[recNum] <
                     dataFile['Move In Date'].values[recNum])):
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

    # Dump the processed data as a pickle file so that we don't have to keep re-doing this
    # --------------------------------------------------------------------------------------------------------
    pickle.dump(dataFile, open(os.path.join(args.pwd, 'data/rentalData.pkl'), 'wb'))

# Histogram the rental durations
# ------------------------------------------------------------------------------------------------------------

if args.plot:
    print 'Making histogram of durations'
    binList = [0.0, 0.1, Overnight, 1.0, 30.0, 60.0, 90.0, 120, 150, 180.0, 210, 240, 270.0, 300.0, 330.0,
               360.0, 390.0, 420.0, 450.0, 480.0, 510.0, 540.0, 570.0, 600.0]
    goodData = dataFile['Rental Duration'].values[dataFile['Not Rented?'].values +
                                                  dataFile['Bad Date'].values +
                                                  dataFile['Not Moved Out?'].values == 0]
    histOut = plt.hist(
        goodData,
        bins=binList, normed=False, facecolor='green', alpha=0.75)
    plt.xlabel(r'Duration (days)')
    plt.ylabel(r'\# of Rentals')
    if args.save:
        plt.savefig('./plots/RentalDurations.jpg', dpi=DPI)
        if args.pdf:
            plt.savefig('./plots/RentalDurations.pdf', dpi=DPI)


# Estimate very simple probabilities for the rental using ratios
# ------------------------------------------------------------------------------------------------------------
if args.ratios:
    print 'Computing simple ratio estimates of probabilities'
    # First find all the unique promotions
    promoList = list(set(dataFile['Promotion Name'].values.tolist()))
    reOrder = [5, 3, 1, 2, 0, 4]
    promoList = [promoList[index] for index in reOrder]
    numPromos = len(promoList)

    # Now do simple counts to get the raw ratios
    countsArray = np.zeros((numPromos, 2), dtype=float)
    for i in xrange(numRecords):
        if dataFile['Bad Date'].values[i] == 0:
            promoNum = promoList.index(dataFile['Promotion Name'].values[i])
            countsArray[promoNum, 1] += 1.0
            if dataFile['Rented?'].values[i] != 0:
                countsArray[promoNum, 0] += 1.0
    # Dictionary - key: promoName; value: [raw likelihood, number of datapoints]
    rentedProbs = dict((promo, [countsArray[counter, 0]/countsArray[counter, 1],
                                int(countsArray[counter, 0]), int(countsArray[counter, 1])])
                       for counter, promo in enumerate(promoList))
    if args.plot:
        plt.clf()
        plt.figure(1, figsize=(fwid, fhgt))
        plt.barh(range(len(rentedProbs)), [value[0] for value in rentedProbs.values()], align='center')
        plt.yticks(range(len(rentedProbs)), rentedProbs.keys())
        plt.ylabel(r'Promotion Type')
        plt.xlabel(r'$\mathrm{Reservation} \rightarrow \mathrm{Rental}$')
        plt.xlim(0.0, 1.0)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        if args.save:
            plt.savefig('./plots/RentalProbabilities.jpg', dpi=DPI)
            if args.pdf:
                plt.savefig('./plots/RentalProbabilities.pdf', dpi=DPI)

# Examine the rate v/s the duration
# ------------------------------------------------------------------------------------------------------------

if args.stop:
    pdb.set_trace()
