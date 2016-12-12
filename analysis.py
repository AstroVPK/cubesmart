import math
import os
import argparse
import cPickle as pickle
import numpy as np
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt
import warnings
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
rentedData = cleanData[(cleanData['CarSpot'] == 0) &
                       (cleanData['Rented'] != 0) &
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
    for i, grp in enumerate(groupList):
        plt.scatter(Foo[grp], Dur[grp],
                    c=bmap.hex_colors[i],
                    marker='.', edgecolors='none',
                    label=r'%s: $\kappa = \$ %3.2f$'%(promoList[i], np.mean(Foo[grp].values)))
    plt.ylim(np.min(rentedData['RentalDuration'].values), np.max(rentedData['RentalDuration'].values))
    plt.xlim(np.min(rentedData['FootRate'].values), np.max(rentedData['FootRate'].values))
    plt.legend()
    plt.ylabel('Rental Duration (days)')
    plt.xlabel('Rate Per Square Foot ($\$/ft^{2}$)')
    if args.save:
        plt.savefig('./plots/FootRateVSDurations.jpg', dpi=args.dpi)
        if args.pdf:
            plt.savefig('./plots/FootRateVSDurations.pdf', dpi=args.dpi)

    NPRate = np.mean(Foo[NP].values)
    FMHORate = np.mean(Foo[FMHO].values)
    TwMHORate = np.mean(Foo[TwMHO].values)
    ThMHORate = np.mean(Foo[ThMHO].values)
    FMFRate = np.mean(Foo[FMF].values)
    TMFRate = np.mean(Foo[TMF].values)

# Use survival analysis to study the rental durations since our data is right-censored
# ------------------------------------------------------------------------------------------------------------


# First, a function to compute the median time at which the renter moves out
def getMedianTime(kmf, order=3):
    if not np.isinf(kmf.median_):
        return kmf.median_
    else:  # Estimate the median duration by fitting the survival function
        warnings.warn('Un-able to estimate median parametrically - using smoothing spline')
        raw = kmf.survival_function_[kmf.survival_function_.columns[0]].values
        index = np.where(raw == raw[-1])[0][0]
        logt = np.log10(kmf.timeline[0: index])
        logy = np.log10(kmf.survival_function_[kmf.survival_function_.columns[0]].values[0: index])
        spl = scipy.interpolate.UnivariateSpline(logy[1:], logt[1:], k=order)
        medianVal = math.pow(10.0, spl(-0.3010299956639812))  # -0.3010299956639812 = log10(0.5)
        return medianVal

if args.survival:
    print 'Performing survival analysis'
    Dur = rentedData['RentalDuration']
    MovedOut = rentedData['MovedOut']
    kmf = lifelines.KaplanMeierFitter()

    # Get the Kaplan-Meier Estimate for some promotion vs no promotion
    kmf.fit(Dur[NP], event_observed=MovedOut[NP], label='No Promotion')
    binaryNoPromoDur = getMedianTime(kmf)
    print 'Median Rental Duration when no promotion was offered: ', binaryNoPromoDur
    if args.plot:
        ax = kmf.plot()
    kmf.fit(Dur[~NP], MovedOut[~NP], label='Some Promotion')
    binaryPromoDur = getMedianTime(kmf)
    print 'Median Rental Duration when some promotion was offered:', binaryPromoDur
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
    NPDur = getMedianTime(kmf)
    print 'Median Rental Duration when "No Promotion" was offered:', NPDur
    if args.plot:
        ax = kmf.plot()
    kmf.fit(Dur[FMF], MovedOut[FMF], label='First Month Free')
    FMFDur = getMedianTime(kmf)
    print 'Median Rental Duration when "First Month Free" was offered:', FMFDur
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[FMHO], MovedOut[FMHO], label='First Month Half Off')
    FMHODur = getMedianTime(kmf)
    print 'Median Rental Duration when "First Month Half Off" was offered:', FMHODur
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[TMF], MovedOut[TMF], label='Two Months Free')
    TwMFDur = getMedianTime(kmf)
    print 'Median Rental Duration when "Two Months Free" was offered:', TwMFDur
    if args.plot:
        kmf.plot(ax=ax)
    kmf.fit(Dur[TwMHO], MovedOut[TwMHO], label='Two Months Half Off')
    TwMHODur = getMedianTime(kmf)
    print 'Median Rental Duration when "Two Months Half Off" was offered:', TwMHODur
    if args.plot:
        kmf.plot(ax=ax)
    if args.plot:
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            fig.savefig('./plots/DurationsSurvival_Breakdown_Overlaid.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/DurationsSurvival_Breakdown_Overlaid.pdf', dpi=args.dpi)
    # Now make one plot with individual subplots showing each case
    fig = plt.figure(2, figsize=(args.width, args.height))
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

    # Lets do a survival regression
    if os.path.isfile(os.path.join(os.path.join(args.pwd, 'data/analysis_survival.pkl'))):
        print 'Reading in pickled survival regression data file...'
        with open(os.path.join(args.pwd, 'data/analysis_survival.pkl'), 'rb') as inFile:
            X = pickle.load(inFile)
            aaf = pickle.load(inFile)
    else:
        print 'Computing survival regression...'
        X = patsy.dmatrix('AccountID + FootRate + PromotionName - 1',
                          rentedData, return_type="dataframe")
        X = X.rename(columns={'PromotionName[NoPromotion]': 'NoPromotion',
                              'PromotionName[FirstMonthFree]': 'FirstMonthFree',
                              'PromotionName[FirstMonthHalfOff]': 'FirstMonthHalfOff',
                              'PromotionName[TwoMonthsFree]': 'TwoMonthsFree',
                              'PromotionName[TwoMonthsHalfOff]': 'TwoMonthsHalfOff'})
        aaf = lifelines.AalenAdditiveFitter(coef_penalizer=1.0, fit_intercept=True)
        X['Dur'] = rentedData['RentalDuration']
        X['MOut'] = rentedData['MovedOut']
        aaf.fit(X[1:], 'Dur', event_col='MOut')
        with open(os.path.join(args.pwd, 'data/analysis_survival.pkl'), 'wb') as outFile:
            pickle.dump(X, outFile)
            pickle.dump(aaf, outFile)

    if args.plot:
        aaf.plot(columns=['NoPromotion', 'FirstMonthFree', 'FirstMonthHalfOff', 'TwoMonthsFree',
                          'TwoMonthsHalfOff', 'FootRate', 'baseline'], ix=slice(0, 500))
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        if args.save:
            fig.savefig('./plots/RegressedCumulativeHazards.jpg', dpi=args.dpi)
            if args.pdf:
                fig.savefig('./plots/RegressedCumulativeHazards.pdf', dpi=args.dpi)

    '''
    delta_expec = np.zeros(numRecords, dtype=float)
    delta_median = np.zeros(numRecords, dtype=float)
    for recNum in xrange(numStored):
        ix = (storageData['AccountID'] == storageData['AccountID'].values[recNum])
        entry = X.ix[ix]
        trueVal = storageData['AccountID'].values[recNum]
        delta_expec[recNum] = trueVal - aaf.predict_expectation(entry)[0].values[0]
        delta_median[recNum] = trueVal - aaf.predict_median(entry)
    storageData['DeltaExpec'] = delta_expec
    storageData['DeltaMedian'] = delta_median
    plt.show()
    pdb.set_trace()
    '''

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
        plt.figure(3, figsize=(args.width, args.height))
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

# Lets do a logistic regression
# ------------------------------------------------------------------------------------------------------------

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


# Now lets compute the revenue curve
# ------------------------------------------------------------------------------------------------------------
def durToMon(dur):
    return int(math.ceil(dur%30.0)) - 3

numSamples = 100
trialFootRate = np.linspace(0.3, 4.0, numSamples)
trialProb = np.zeros((5, numSamples))
trialRevenue = np.zeros((5, numSamples))
currRevenue = np.zeros(5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in xrange(numSamples):
        trialProb[0, i] = model.predict_proba(np.array([1.0, 0.0, 0.0, 0.0, 0.0, trialFootRate[i]]))[0][1]
        trialRevenue[0, i] = (3.0 + durToMon(NPDur))*trialProb[0, i]*trialFootRate[i]
        trialProb[1, i] = model.predict_proba(np.array([1.0, 1.0, 0.0, 0.0, 0.0, trialFootRate[i]]))[0][1]
        trialRevenue[1, i] = (2.0 + durToMon(FMFDur))*trialProb[1, i]*trialFootRate[i]
        trialProb[2, i] = model.predict_proba(np.array([1.0, 0.0, 1.0, 0.0, 0.0, trialFootRate[i]]))[0][1]
        trialRevenue[2, i] = (2.5 + durToMon(FMHODur))*trialProb[2, i]*trialFootRate[i]
        trialProb[3, i] = model.predict_proba(np.array([1.0, 0.0, 0.0, 1.0, 0.0, trialFootRate[i]]))[0][1]
        trialRevenue[3, i] = (1.0 + durToMon(TwMFDur))*trialProb[3, i]*trialFootRate[i]
        trialProb[4, i] = model.predict_proba(np.array([1.0, 0.0, 0.0, 0.0, 1.0, trialFootRate[i]]))[0][1]
        trialRevenue[4, i] = (1.5 + durToMon(TwMHODur))*trialProb[4, i]*trialFootRate[i]
    currRevenue[0] = (3.0 + durToMon(NPDur))*model.predict_proba(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, NPRate]))[0][1]*NPRate
    print 'Current revenue No Promotion: ', currRevenue[0]
    currRevenue[1] = (2.0 + durToMon(FMFDur))*model.predict_proba(
        np.array([1.0, 1.0, 0.0, 0.0, 0.0, FMFRate]))[0][1]*FMFRate
    print 'Current revenue First Month Free: ', currRevenue[1]
    currRevenue[2] = (2.5 + durToMon(FMHODur))*model.predict_proba(
        np.array([1.0, 0.0, 1.0, 0.0, 0.0, FMHORate]))[0][1]*FMHORate
    print 'Current revenue First Month Half Off: ', currRevenue[2]
    currRevenue[3] = (1.0 + durToMon(TwMFDur))*model.predict_proba(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, TMFRate]))[0][1]*TMFRate
    print 'Current revenue Two Months Free: ', currRevenue[3]
    currRevenue[4] = (2.0 + durToMon(TwMHODur))*model.predict_proba(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, TwMHORate]))[0][1]*TwMHORate
    print 'Current revenue Two Months Half Off: ', currRevenue[4]
if plt.plot:
    plt.figure(4, figsize=(args.width, args.height))
    plt.clf()
    plt.plot(trialFootRate[:], trialProb[0, :], c='#e41a1c', label='No Promotion')
    plt.plot(trialFootRate[:], trialProb[1, :], c='#377eb8', label='First Month Free')
    plt.plot(trialFootRate[:], trialProb[2, :], c='#4daf4a', label='First Month Half Off')
    plt.plot(trialFootRate[:], trialProb[3, :], c='#984ea3', label='Two Months Free')
    plt.plot(trialFootRate[:], trialProb[4, :], c='#ff7f00', label='Two Months Half Off')
plt.legend()
plt.xlabel('$\log_{10}$ Rate Per Square Foot ($\$/ft^{2}$)')
plt.ylabel('Likelihood')
if args.save:
    plt.savefig('./plots/ProbabilityCurve.jpg', dpi=args.dpi)
    if args.pdf:
        plt.savefig('./plots/ProbabilityCurve.pdf', dpi=args.dpi)
if plt.plot:
    plt.figure(5, figsize=(args.width, args.height))
    plt.clf()

    plt.plot(trialFootRate[:], trialRevenue[0, :], c='#e41a1c', label='No Promotion')
    plt.axvline(x=trialFootRate[np.where(trialRevenue[0, :] == np.max(trialRevenue[0, :]))],
                ls='dashed', c='#e41a1c', label='No Promotion Max Revenue Rate')
    plt.axvline(x=NPRate, ls='dotted', c='#e41a1c',
                label='No Promotion Avg. Rate')

    plt.plot(trialFootRate[:], trialRevenue[1, :], c='#377eb8', label='First Month Free')
    plt.axvline(x=trialFootRate[np.where(trialRevenue[1, :] == np.max(trialRevenue[1, :]))],
                ls='dashed', c='#377eb8', label='First Month Free Max Revenue Rate')
    plt.axvline(x=FMFRate, ls='dotted', c='#377eb8',
                label='First Month Free Avg. Rate')

    plt.plot(trialFootRate[:], trialRevenue[2, :], c='#4daf4a', label='First Month Half Off')
    plt.axvline(x=trialFootRate[np.where(trialRevenue[2, :] == np.max(trialRevenue[2, :]))],
                ls='dashed', c='#4daf4a', label='First Month Half Off Max Revenue Rate')
    plt.axvline(x=FMHORate, ls='dotted', c='#4daf4a',
                label='First Month Half Off Avg. Rate')

    plt.plot(trialFootRate[:], trialRevenue[3, :], c='#984ea3', label='Two Months Free')
    plt.axvline(x=trialFootRate[np.where(trialRevenue[3, :] == np.max(trialRevenue[3, :]))],
                ls='dashed', c='#984ea3', label='Two Months Free Max Revenue Rate')
    plt.axvline(x=TMFRate, ls='dotted', c='#984ea3',
                label='Two Months Free Avg. Rate')

    plt.plot(trialFootRate[:], trialRevenue[4, :], c='#ff7f00', label='Two Months Half Off')
    plt.axvline(x=trialFootRate[np.where(trialRevenue[4, :] == np.max(trialRevenue[4, :]))],
                ls='dashed', c='#ff7f00', label='Two Months Half Off Max Revenue Rate')
    plt.axvline(x=TwMHORate, ls='dotted', c='#ff7f00',
                label='Two Months Half Off Avg. Rate')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.0)
plt.xlabel('Rate Per Square Foot ($\$/ft^{2}$)')
plt.ylabel('Revenue ($\$/ft^{2}$)')
if args.save:
    plt.savefig('./plots/RevenueCurve.jpg', dpi=args.dpi)
    if args.pdf:
        plt.savefig('./plots/RevenueCurve.pdf', dpi=args.dpi)

if args.stop:
    pdb.set_trace()
