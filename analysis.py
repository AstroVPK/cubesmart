import math
import os
import argparse
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import patsy
import lifelines
import lifelines.statistics
import sklearn.linear_model
import brewer2mpl

import mpl_settings  # Fancy formatting rules for plots.

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
parser.add_argument('-fsize', '--fontsize', type=int, default=12, help=r'Fontsize?')
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
parser.add_argument('--suspicious', dest='suspicious', action='store_true', help=r'Individual histogram bins\
    for very short duration rentals?')
parser.add_argument('--no-suspicious', dest='suspicious', action='store_false', help=r'No individual\
    histogram bins for very short duration rentals?')
parser.set_defaults(suspicious=False)
args = parser.parse_args()

mpl_settings.set_plot_params(useTex=True, fontsize=args.fontsize)  # Make plots look good.

if args.ion:
    plt.ion()

# Read in the data as a pandas DataFrame
# ------------------------------------------------------------------------------------------------------------

if os.path.isfile(os.path.join(os.path.join(args.pwd, 'data/analysis.pkl'))):
    print 'Reading in pickled data file...'
    with open(os.path.join(args.pwd, 'data/analysis.pkl'), 'rb') as inFile:
        cleanData = pickle.load(inFile)
        promoList = pickle.load(inFile)
    numRecords = cleanData['AccountID'].count()
    lastRec = numRecords - 1
    numPromos = len(promoList)
else:
    print 'Reading in original excel file...'
    dataFile = pd.read_excel(os.path.join(args.pwd, 'data/TT_Dataset_for_Data_Scientist_Candidate.xlsx'),
                             sheetname=1,
                             header=0)
    dataFile = dataFile.rename(index=str, columns={u'Account ID': u'AccountID', u'Rented?': u'Rented',
                                                   u'Move In Date': u'MoveInDate',
                                                   u'Move Out Date': u'MoveOutDate',
                                                   u'Promotion Name': u'PromotionName',
                                                   u'Previously Used Storage?': u'PreviouslyUsedStorage',
                                                   u'Purpose?': u'Purpose',
                                                   u'Storage Reason': u'StorageReason'})
    numRecords = dataFile['AccountID'].count()
    lastRec = numRecords - 1

    # Clean the data to remove bad entries etc...
    # --------------------------------------------------------------------------------------------------------

    # First make a list of the unique promotions
    promoList = ['NoPromotion', 'FirstMonthHalfOff', 'TwoMonthsHalfOff', 'ThreeMonthsHalfOff',
                 'FirstMonthFree', 'TwoMonthsFree']
    numPromos = len(promoList)

    # Now clean and compute ancillary data
    badDate = list()
    durationArr = np.zeros(numRecords, dtype=float)
    movedOutArr = np.zeros(numRecords, dtype=bool)
    footRateArr = np.zeros(numRecords, dtype=float)
    promoValArr = np.zeros(numRecords, dtype=float)
    carSpotArr = np.zeros(numRecords, dtype=bool)
    infoStr = 'ID: %d; Rented %r; ReserveDate: %s; MoveInDate: %s; MoveOutDate %s;\
               Duration (hrs): %f; Promo: %s'
    for recNum in xrange(numRecords):
        # Convert nan to No Promotion
        if pd.isnull(dataFile['PromotionName'].values[recNum]):
            dataFile['PromotionName'].values[recNum] = u'NoPromotion'
        if dataFile['PromotionName'].values[recNum] == u'Two Months Half Off':
            dataFile['PromotionName'].values[recNum] = u'TwoMonthsHalfOff'
        if dataFile['PromotionName'].values[recNum] == u'Two Months Free':
            dataFile['PromotionName'].values[recNum] = u'TwoMonthsFree'
        if dataFile['PromotionName'].values[recNum] == u'First Month Free':
            dataFile['PromotionName'].values[recNum] = u'FirstMonthFree'
        if dataFile['PromotionName'].values[recNum] == u'Three Months Half off':
            dataFile['PromotionName'].values[recNum] = u'ThreeMonthsHalfOff'
        if dataFile['PromotionName'].values[recNum] == u'First Month Half Off':
            dataFile['PromotionName'].values[recNum] = u'FirstMonthHalfOff'
        # Compute the promotion
        promoValArr[recNum] = float(promoList.index(dataFile['PromotionName'].values[recNum]))
        # Convert 0 SquareFeet (car spot) to 1.0 & compute rate/foot
        if dataFile['SquareFeet'].values[recNum] == 0:
            carSpotArr[recNum] = True
            footRateArr[recNum] = dataFile['RentRate'].values[recNum]
        else:
            footRateArr[recNum] = dataFile['RentRate'].values[recNum]/dataFile['SquareFeet'].values[recNum]
        # Convert all dates to pd.Timestamp for convenience
        dataFile['ReserveDate'].values[recNum] = pd.Timestamp(dataFile['ReserveDate'].values[recNum])
        dataFile['MoveInDate'].values[recNum] = pd.Timestamp(dataFile['MoveInDate'].values[recNum])
        if not pd.isnull(dataFile['MoveOutDate'].values[recNum]):
            dataFile['MoveOutDate'].values[recNum] = pd.Timestamp(
                dataFile['MoveOutDate'].values[recNum])

        # Check for bad reserve date.
        if ((dataFile['ReserveDate'].values[recNum] < dataFile['MoveInDate'].values[5]) or
                (dataFile['ReserveDate'].values[recNum] > dataFile['MoveInDate'].values[lastRec])):
            badDate.append(recNum)
        # Check for bad move in date
        if ((dataFile['MoveInDate'].values[recNum] < dataFile['MoveInDate'].values[5]) or
                (dataFile['MoveInDate'].values[recNum] > dataFile['MoveInDate'].values[lastRec])):
            badDate.append(recNum)
        # If rented, check if moved out and compute duration of stay.
        if dataFile['Rented'].values[recNum] == 1:  # Was the unit rented?
            if not pd.isnull(dataFile['MoveOutDate'].values[recNum]):  # Do we know the move out date?
                # Check for bad move out date.
                if ((dataFile['MoveOutDate'].values[recNum] <
                     dataFile['MoveInDate'].values[5]) or
                    (dataFile['MoveOutDate'].values[recNum] >
                     dataFile['MoveInDate'].values[lastRec]) or
                    (dataFile['MoveOutDate'].values[recNum] <
                     dataFile['MoveInDate'].values[recNum])):
                    badDate.append(recNum)
                # Compute rental duration = move out date - move in date
                duration = dataFile['MoveOutDate'].values[recNum] - dataFile['MoveInDate'].values[recNum]
                movedOut = True
            else:
                # Rental still occupied! Compute rental duration = last reserve date - move in date
                duration = dataFile['MoveInDate'].values[lastRec] - dataFile['MoveInDate'].values[recNum]
                movedOut = False
            durationArr[recNum] = duration/np.timedelta64(1, 'D')
            # Really we should check the actual timestamps but this is much simpler.
            if durationArr[recNum] < Overnight:
                print infoStr%(dataFile['AccountID'].values[recNum],
                               dataFile['Rented'].values[recNum],
                               str(dataFile['ReserveDate'].values[recNum]),
                               str(dataFile['MoveInDate'].values[recNum]),
                               str(dataFile['MoveOutDate'].values[recNum]),
                               durationArr[recNum]*24,
                               dataFile['PromotionName'].values[recNum])
            movedOutArr[recNum] = movedOut
    dataFile['RentalDuration'] = durationArr
    dataFile['MovedOut'] = movedOutArr
    dataFile['FootRate'] = footRateArr
    dataFile['Promotion'] = promoValArr
    dataFile['CarSpot'] = carSpotArr
    badDate = list(set(badDate))
    cleanData = dataFile.drop(dataFile.index[badDate])
    numRecords = cleanData['AccountID'].count()
    lastRec = numRecords - 1

    # Dump the processed data as a pickle file so that we don't have to keep re-doing this
    # --------------------------------------------------------------------------------------------------------
    with open(os.path.join(args.pwd, 'data/analysis.pkl'), 'wb') as outFile:
        pickle.dump(cleanData, outFile)
        pickle.dump(promoList, outFile)

# Get rid of car spot rentals for convenience
storageData = cleanData[(cleanData['CarSpot'] == 0) &
                        (cleanData['PromotionName'] != 'ThreeMonthsHalfOff')]
numStored = storageData['AccountID'].count()

# Create a data frame of just the rentals for convenience
rentedData = cleanData[(cleanData['Rented'] != 0) &
                       (cleanData['PromotionName'] != 'ThreeMonthsHalfOff')]
numRented = rentedData['AccountID'].count()

# Create a data frame of just the rentals that have already moved out for convenience
movedOutData = rentedData[(rentedData['MovedOut'] != 0)]
numMovedOut = movedOutData['AccountID'].count()

# Create some groups to split things up by
promoOffered = rentedData['PromotionName']
NP = (promoOffered == 'NoPromotion')
FMHO = (promoOffered == 'FirstMonthHalfOff')
TwMHO = (promoOffered == 'TwoMonthsHalfOff')
ThMHO = (promoOffered == 'TwoMonthsHalfOff')
FMF = (promoOffered == 'FirstMonthFree')
TMF = (promoOffered == 'TwoMonthsFree')
groupList = [NP, FMHO, TwMHO, ThMHO, FMF, TMF]

# Examine the rate v/s the duration
# ------------------------------------------------------------------------------------------------------------

if args.plot:
    bmap = brewer2mpl.get_map('Dark2', 'Qualitative', 6)
    Dur = movedOutData['RentalDuration']
    Foo = movedOutData['FootRate']
    plt.clf()
    plt.figure(1, figsize=(args.width, args.height))
    for i, grp in enumerate(reversed(groupList)):
        plt.scatter(Dur[grp], np.log10(Foo[grp]),
                    c=bmap.hex_colors[i],
                    marker='.', edgecolors='none', label=promoList[i])
    plt.xlim(np.min(rentedData['RentalDuration'].values), np.max(rentedData['RentalDuration'].values))
    plt.ylim(np.min(np.log10(rentedData['FootRate'].values)), np.max(np.log10(rentedData['FootRate'].values)))
    plt.legend()
    plt.xlabel('Rental Duration (days)')
    plt.ylabel('$\log_{10}$ Rate Per Square Foot ($\$/ft^{2}$)')
    if args.save:
        plt.savefig('./plots/FootRateVSDurations.jpg', dpi=args.dpi)
        if args.pdf:
            plt.savefig('./plots/FootRateVSDurations.pdf', dpi=args.dpi)

# Use survival analysis to study the rental durations since our data is right-censored
# ------------------------------------------------------------------------------------------------------------

if args.survival:
    print 'Performing survival analysis'
    Dur = rentedData['RentalDuration']
    MovedOut = rentedData['MovedOut']
    kmf = lifelines.KaplanMeierFitter()

    # Get the Kaplan-Meier Estimate for some promotion vs no promotion
    kmf.fit(Dur[NP], event_observed=MovedOut[NP], label='No Promotion')
    print 'Median Rental Duration when no promotion" was offered:', kmf.median_
    if args.plot:
        ax = kmf.plot()
    kmf.fit(Dur[~NP], MovedOut[~NP], label='Some Promotion')
    print 'Median Rental Duration when some promotion was offered:', kmf.median_
    if args.plot:
        kmf.plot(ax=ax)
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            fig.savefig('./plots/DurationsSurvival_YN.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/DurationsSurvival_YN.pdf', dpi=args.dpi)
    # Does the promotion affect the rental duration?
    results = lifelines.statistics.logrank_test(Dur[NP], Dur[~NP], MovedOut[NP], MovedOut[~NP], alpha=0.99)
    results.print_summary()

    # Now get the Kaplan-Meier Estimate for each promotion
    # First make one plot showing all cases
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
            fig.savefig('./plots/DurationsSurvival_Breakdown_Overlaid.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/DurationsSurvival_Breakdown_Overlaid.pdf', dpi=args.dpi)
    # Now make one plot with individual subplots showing each case
    fig = plt.figure(1, figsize=(args.width, args.height))
    fig.clf()
    ax = fig.add_subplot(111)    # The big subplot
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    for i, promoType in enumerate(promoList):
        if promoType == 'ThreeMonthsHalfOff':
            continue
        axS = fig.add_subplot(2, 3, i + 1)
        ix = rentedData['PromotionName'] == promoType
        kmf.fit(Dur[ix], MovedOut[ix], timeline=np.linspace(0.0, 500.0, 5000), label=promoType)
        if args.plot:
            kmf.plot(ax=axS, legend=False)
            plt.title(promoType)
            plt.xlim(0.0, 500.0)
            plt.ylim(0.0, 1.0)
    if args.plot:
        plt.tight_layout()
        ax.set_ylabel('Frac. still renting after $t$ months')
        if args.save:
            fig.savefig('./plots/DurationsSurvival_Breakdown_Panelled.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/DurationsSurvival_Breakdown_Panelled.pdf', dpi=args.dpi)

    # Now lets try to estimate the hazard function & cumulative hazard function
    bandWidth = 5.0
    naf = lifelines.NelsonAalenFitter()
    naf.fit(Dur[NP], event_observed=MovedOut[NP], label="No Promotion")
    if args.plot:
        plt.clf()
        axC = naf.plot()
        figC = plt.gcf()
        axD = naf.plot_hazard(bandwidth=bandWidth)
        figD = plt.gcf()
    naf.fit(Dur[~NP], event_observed=MovedOut[~NP], label="Promotion Offered")
    if args.plot:
        naf.plot(ax=axC)
        plt.title("Cumulative hazard function of moving out")
        if args.save:
            figC.savefig('./plots/CumulativeHazard.jpg', dpi=args.dpi)
            if args.pdf:
                figC.savefig('./plots/CumulativeHazard.pdf', dpi=args.dpi)
        naf.plot_hazard(ax=axD, bandwidth=bandWidth)
        plt.title("Hazard function of moving out")
        if args.save:
            figD.savefig('./plots/Hazard.jpg', dpi=args.dpi)
            if args.pdf:
                figD.savefig('./plots/Hazard.pdf', dpi=args.dpi)

    # Finally, make a histogram of the rental durations in the case where the renter has checked out.
    if args.plot:
        plt.clf()
        histData = cleanData[(cleanData['Rented'] != 0) & (cleanData['MovedOut'] != 0)]
        if args.suspicious:
            binList = [math.log10(a)
                       for a in [np.min(histData['RentalDuration'].values),
                                 Overnight, 1.0] + [float(i) for i in xrange(30, 630, 30)]]
            histOut = plt.hist(
                np.log10(histData['RentalDuration'].values), bins=binList,
                normed=False, facecolor='green', alpha=0.75)
            plt.xlabel(r'$\log_{10}$ Duration (days)')
        else:
            binList = [float(a) for a in xrange(0, 630, 30)]
            histOut = plt.hist(
                histData['RentalDuration'].values, bins=binList,
                normed=False, facecolor='green', alpha=0.75)
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
        promoNum = promoList.index(cleanData['PromotionName'].values[i])
        countsArray[promoNum, 1] += 1.0
        if cleanData['Rented'].values[i] != 0:
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

# Now lets do a logistic regression
# ------------------------------------------------------------------------------------------------------------

y, X = patsy.dmatrices('Rented ~ FootRate + RentalDuration + C(PromotionName, Treatment("NoPromotion"))',
                       storageData, return_type="dataframe")
X = X.rename(columns={'C(PromotionName, Treatment("NoPromotion"))[T.FirstMonthFree]': 'FirstMonthFree',
                      'C(PromotionName, Treatment("NoPromotion"))[T.FirstMonthHalfOff]': 'FirstMonthHalfOff',
                      'C(PromotionName, Treatment("NoPromotion"))[T.TwoMonthsFree]': 'TwoMonthsFree',
                      'C(PromotionName, Treatment("NoPromotion"))[T.TwoMonthsHalfOff]': 'TwoMonthsHalfOff'})
y = y['Rented[True]'].values

model = sklearn.linear_model.LogisticRegression()
model = model.fit(X, y)
print model.score(X, y)
print y.mean()
print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

# Now lets do a logistic regression again but do promotion vs no promotion
# ------------------------------------------------------------------------------------------------------------
'''
y, X = patsy.dmatrices('Rented ~ FootRate + C(PromotionName, Treatment("NoPromotion"))',
                       storageData, return_type="dataframe")
X = X.rename(columns={'C(PromotionName, Treatment("NoPromotion"))[T.FirstMonthFree]': 'FirstMonthFree',
                      'C(PromotionName, Treatment("NoPromotion"))[T.FirstMonthHalfOff]': 'FirstMonthHalfOff',
                      'C(PromotionName, Treatment("NoPromotion"))[T.TwoMonthsFree]': 'TwoMonthsFree',
                      'C(PromotionName, Treatment("NoPromotion"))[T.TwoMonthsHalfOff]': 'TwoMonthsHalfOff'})
y = y['Rented[True]'].values

model = sklearn.linear_model.LogisticRegression()
model = model.fit(X, y)
print model.score(X, y)
print y.mean()
print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
'''
if args.stop:
    pdb.set_trace()
