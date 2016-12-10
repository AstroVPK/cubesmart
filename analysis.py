import math
import os
import argparse
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import lifelines

import mpl_settings  # Fancy formatting rules for plots.

mpl_settings.set_plot_params(useTex=True)  # Make plots look good.
rows, columns = os.popen('stty size', 'r').read().split()
np.set_printoptions(linewidth=int(columns) - 2)

Overnight = 8.0/24.0  # Simplistic duration of day that we want to consider as a single night rental.

parser = argparse.ArgumentParser()
parser.add_argument('-pwd', '--pwd', type=str, default=os.environ['PWD'], help=r'Path to working directory')
parser.add_argument('--plot', dest='plot', action='store_true', help=r'Make & show plot?')
parser.add_argument('--no-plot', dest='plot', action='store_false', help=r'Do not make & show plot?')
parser.set_defaults(plot=True)
parser.add_argument('-dpi', '--dpi', type=int, default=300, help=r'Dots Per Inch to save figure at?')
parser.add_argument('-wdth', '--width', type=int, default=16, help=r'Figure width?')
parser.add_argument('-hght', '--height', type=int, default=10, help=r'Figure height?')
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
parser.add_argument('--survival', dest='survival', action='store_true', help=r'Perform survival analysis?')
parser.add_argument('--no-survival', dest='survival', action='store_false', help=r'Do not perform survival\
                     analysis?')
parser.set_defaults(survival=True)
parser.add_argument('--ratios', dest='ratios', action='store_true', help=r'Compute simple ratios?')
parser.add_argument('--no-ratios', dest='ratios', action='store_false', help=r'Do not compute simple ratios?')
parser.set_defaults(ratios=True)
args = parser.parse_args()

if args.ion:
    plt.ion()

# Read in the data as a pandas DataFrame
# ------------------------------------------------------------------------------------------------------------

if os.path.isfile(os.path.join(os.path.join(args.pwd, 'data/analysis.pkl'))):
    print 'Reading in pickled data file...'
    with open(os.path.join(args.pwd, 'data/analysis.pkl'), 'rb') as inFile:
        cleanData = pickle.load(inFile)
        promoList = pickle.load(inFile)
    numRecords = cleanData['Account ID'].count()
    lastRec = numRecords - 1
    numPromos = len(promoList)
else:
    print 'Reading in original excel file...'
    dataFile = pd.read_excel(os.path.join(args.pwd, 'data/TT_Dataset_for_Data_Scientist_Candidate.xlsx'),
                             sheetname=1,
                             header=0)
    numRecords = dataFile['Account ID'].count()
    lastRec = numRecords - 1

    # Now clean the data to remove bad entries etc...
    # --------------------------------------------------------------------------------------------------------

    badDate = list()
    durationArr = np.zeros(numRecords, dtype=float)
    movedOutArr = np.zeros(numRecords, dtype=bool)
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
        if ((dataFile['ReserveDate'].values[recNum] < dataFile['Move In Date'].values[5]) or
                (dataFile['ReserveDate'].values[recNum] > dataFile['Move In Date'].values[lastRec])):
            badDate.append(recNum)
        # Check for bad move in date
        if ((dataFile['Move In Date'].values[recNum] < dataFile['Move In Date'].values[5]) or
                (dataFile['Move In Date'].values[recNum] > dataFile['Move In Date'].values[lastRec])):
            badDate.append(recNum)
        # If rented, check if moved out and compute duration of stay.
        if dataFile['Rented?'].values[recNum]:  # Was the unit rented?
            if not pd.isnull(dataFile['Move Out Date'].values[recNum]):  # Do we know the move out date?
                # Check for bad move out date.
                if ((dataFile['Move Out Date'].values[recNum] <
                     dataFile['Move In Date'].values[5]) or
                    (dataFile['Move Out Date'].values[recNum] >
                     dataFile['Move In Date'].values[lastRec]) or
                    (dataFile['Move Out Date'].values[recNum] <
                     dataFile['Move In Date'].values[recNum])):
                    badDate.append(recNum)
                # Compute rental duration = move out date - move in date
                duration = dataFile['Move Out Date'].values[recNum] - dataFile['Move In Date'].values[recNum]
                movedOut = True
            else:
                # Rental still occupied! Compute rental duration = last reserve date - move in date
                duration = dataFile['Move In Date'].values[lastRec] - dataFile['Move In Date'].values[recNum]
                movedOut = False
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
            movedOutArr[recNum] = movedOut
    dataFile['Rental Duration'] = durationArr
    dataFile['Moved Out?'] = movedOutArr
    badDate = list(set(badDate))
    cleanData = dataFile.drop(dataFile.index[badDate])
    numRecords = cleanData['Account ID'].count()
    lastRec = numRecords - 1

    # Find all the unique promotions
    promoList = list(set(cleanData['Promotion Name'].values.tolist()))
    reOrder = [5, 3, 1, 2, 0, 4]
    promoList = [promoList[index] for index in reOrder]
    numPromos = len(promoList)

    # Dump the processed data as a pickle file so that we don't have to keep re-doing this
    # --------------------------------------------------------------------------------------------------------
    with open(os.path.join(args.pwd, 'data/analysis.pkl'), 'wb') as outFile:
        pickle.dump(cleanData, outFile)
        pickle.dump(promoList, outFile)

# Use suyrvival analysis to study the rental durations since our data is right-censored
# ------------------------------------------------------------------------------------------------------------

if args.survival:
    print 'Making histogram of durations'
    binList = [0.0, 0.1, Overnight, 1.0] + [float(i) for i in xrange(30, 630, 30)]
    rentedData = cleanData[(cleanData['Rented?'] != 0)]
    numRented = rentedData['Account ID'].count()
    Dur = rentedData['Rental Duration']
    MovedOut = rentedData['Moved Out?']

    # Begin by getting Kaplan-Meier Estimate for the whole population
    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(Dur, event_observed=MovedOut)
    print 'Overall Median Rental Duration:', kmf.median_
    if args.plot:
        axAll = kmf.plot()
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            fig.savefig('./plots/All_RentalDurationsSurvival.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/All_RentalDurationsSurvival.pdf', dpi=args.dpi)
    promoOffered = rentedData['Promotion Name']
    NP = (promoOffered == 'No Promotion')
    TMF = (promoOffered == 'Two Months Free')
    FMF = (promoOffered == 'First Month Free')
    TwMHO = (promoOffered == 'Two Months Half Off')
    FMHO = (promoOffered == 'First Month Half Off')
    kmf.fit(Dur[NP], MovedOut[NP], label='No Promotion')
    print 'Median Rental Duration when "No Promotion" was offered:', kmf.median_
    if args.plot:
        ax = kmf.plot()
    kmf.fit(Dur[TMF], MovedOut[TMF], label='Two Months Free')
    print 'Median Rental Duration when "Two Months Free" was offered:', kmf.median_
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[FMF], MovedOut[FMF], label='First Month Free')
    print 'Median Rental Duration when "First Month Free" was offered:', kmf.median_
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[TwMHO], MovedOut[TwMHO], label='Two Months Half Off')
    print 'Median Rental Duration when "Two Months Half Off" was offered:', kmf.median_
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[FMHO], MovedOut[FMHO], label='First Month Half Off')
    print 'Median Rental Duration when "First Month Half Off" was offered:', kmf.median_
    if args.plot:
        kmf.plot(ax=ax)
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            fig.savefig('./plots/All_RentalDurationsSurvival_Overlaid.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/All_RentalDurationsSurvival_Overlaid.pdf', dpi=args.dpi)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)    # The big subplot
    for i, promoType in enumerate(promoList):
        if promoType == 'Three Months Half off':
            continue
        ax = plt.subplot(2, 3, i + 1)
        ix = rentedData['Promotion Name'] == promoType
        kmf.fit(Dur[ix], MovedOut[ix], timeline=np.linspace(0.0, 500.0, 5000), label=promoType)
        if args.plot:
            kmf.plot(ax=ax, legend=False)
            plt.title(promoType)
            plt.xlim(0.0, 500.0)
            plt.ylim(0.0, 1.0)
    if args.plot:
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        plt.ylabel('Frac. still renting after $t$ months')
        if args.save:
            fig.savefig('./plots/All_RentalDurationsSurvival_Panelled.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/All_RentalDurationsSurvival_Panelled.pdf', dpi=args.dpi)

    if args.plot:
        histData = cleanData[(cleanData['Rented?'] != 0) & (cleanData['Moved Out?'] != 0)]
        histOut = plt.hist(
            histData['Rental Duration'].values,
            bins=binList, normed=False, facecolor='green', alpha=0.75)
        plt.xlabel(r'Duration (days)')
        plt.ylabel(r'\# of Rentals')
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            plt.savefig('./plots/RentalDurations.jpg', dpi=args.dpi)
            if args.pdf:
                plt.savefig('./plots/RentalDurations.pdf', dpi=args.dpi)


# Estimate very simple probabilities for the rental using ratios
# ------------------------------------------------------------------------------------------------------------
if args.ratios:
    print 'Computing simple ratio estimates of probabilities'

    # Do simple counts to get the raw ratios
    countsArray = np.zeros((numPromos, 2), dtype=float)
    for i in xrange(numRecords):
        promoNum = promoList.index(cleanData['Promotion Name'].values[i])
        countsArray[promoNum, 1] += 1.0
        if cleanData['Rented?'].values[i] != 0:
            countsArray[promoNum, 0] += 1.0
    # Dictionary - key: promoName; value: [raw likelihood, number of datapoints]
    rentedProbs = dict((promo, [countsArray[counter, 0]/countsArray[counter, 1],
                                int(countsArray[counter, 0]), int(countsArray[counter, 1])])
                       for counter, promo in enumerate(promoList))
    if args.plot:
        plt.clf()
        plt.figure(1, figsize=(args.width, args.height))
        plt.barh(range(len(rentedProbs)), [value[0] for value in rentedProbs.values()], align='center')
        plt.yticks(range(len(rentedProbs)), rentedProbs.keys())
        plt.ylabel(r'Promotion Type')
        plt.xlabel(r'$\mathrm{Reservation} \rightarrow \mathrm{Rental}$')
        plt.xlim(0.0, 1.0)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        if args.save:
            plt.savefig('./plots/RentalProbabilities.jpg', dpi=args.dpi)
            if args.pdf:
                plt.savefig('./plots/RentalProbabilities.pdf', dpi=args.dpi)

# Examine the rate v/s the duration
# ------------------------------------------------------------------------------------------------------------

if args.plot:
    plt.clf()
    plt.figure(1, figsize=(args.width, args.height))
    plt.scatter(cleanData['Rental Duration'].values, cleanData['RentRate'].values)

if args.stop:
    pdb.set_trace()
