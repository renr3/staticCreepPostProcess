#Import library
import csv
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pickle

#To search for CSV files automatically
import os
import glob

class testSpecimen():
    """
    A class that defines all the parameters of a test specimen
    """
    def __init__(self, specimenID, acquisitionSystem, channelsLVDT, baseLength, appliedStress, channelsName, lineColor, lineStyleForPlot, uncalibratedValues=False, absoluteValues=False):
        """
        Initialization of class
        Parameters:
            - specimenID: str
                Defines the identification used to refer to this specimen
            - acquisitionSystem: str
                Defines the acquisition system used to monitor this specimen
            - channelsLVDT: list of int
                Defines the columns in the text file associated to the LVDTs used to monitor this specimen.
                To know which numbers identify the LVDTs of a specimen, you need to know the LVDTs of the specimens and their position in the result file
                This assumes that for a single experiment, the test files always follow the same order of channels in the logged data.
            - baseLength: float
                Defines the base length of the LVDTs of the specimen, in mm
            - appliedStress: float
                Defines the applied load in the specimen, in MPa
            - channelsName: list of str
                Defines the identification used to refer to each LVDT of the specimen. 
                List is ordered, and must follow the same order of channelsLVDT
            - lineColor: list of ()
                Defines the line colors used to plot each LVDT data
            - lineStyleForPlot: list of ()
                Defines the line style used to plot each LVDT data
            - uncalibratedValues: optional, boolean
                For National system, choose between working with calibrated or uncalibrated values
            - absoluteValues: optional, boolean
                For National system, choose between automatically zero-shiftting in relation to the first measurement or using absolute measurement data
                This is useful for channels that we need to unmix

        Aditional class attributes:
            - displacementData: dict of list
                Initiated as an empty list to be populated with displacement data read from experiment result files
            - strainData: dict of list
                Computed from displacement data, holds the strain series for each LVDT
            - timeData: list of datetime object
                Initiated as an empty list to be populated with time data from experiment result files
            - startTimeOfTest: datetime object
                Datetime object that contains the start time of the test, considering the beginning of the creep system data
            - plotTime: list of float
                List to be derived from timeData, but containing the number of seconds elapsed since the beggining of the test
            - seriesStartTime: list
                Initiated as an empty list to be populated with the start time, in datetime object, of each series composing the data (data may be composed by several series, each in a different test file) read from experiment result files
            - creepStartTime: float
                Time in which creep has started, in seconds. Taken from the compliance curve, in which 0 seconds coincides with seriesStartTime.
            - specificCreepTimeData: list of float
                Derived from plotTime, this time series is the number of days elpased since beginning of creep, as defined by self.creepStartTime
            - specificCreep: dict of lists
                Derived from self.complianceData, it is the specific compliance (not considering the initial deformation, also identified as elastic deformation, as defined by creepStartTime)
            - releaseCorrectionList: dict of lists
                Allows correcting for releases/slips that might occur during testing. They need to be manualy and visually detected in each test file.
                Check method addReleaseCorrection() for further understanding on how this attribute works
        """
        #Attributes defined in class instantiation
        self.specimenID=specimenID
        self.acquisitionSystem=acquisitionSystem
        self.channelsLVDT=channelsLVDT
        self.baseLength=baseLength
        self.appliedStress=appliedStress
        self.channelsName=channelsName
        self.lineColor=lineColor
        self.lineStyleForPlot=lineStyleForPlot
        self.uncalibratedValues=uncalibratedValues
        self.absoluteValues=absoluteValues

        #Additional class attributes
        self.displacementData={channel:[] for channel in channelsName}
        self.strainData={channel:[] for channel in channelsName}
        self.complianceData={channel:[] for channel in channelsName}
        self.timeData=[]
        self.startTimeOfTest=None
        self.plotTime=[]
        self.seriesStartTime=[]

        self.creepStartTime=0
        self.specificCreepTimeData=[]
        self.specificCreep={channel:[] for channel in channelsName}

        self.releaseCorrectionList={}

    def setStartOfTest(self, startDateOfTest):
        """
        This function sets the beggining of the test, shifting all experimental data so the start of the data associated to this specimen coincides with startDateOfTest
        It also sets the attribute self.startTimeOfTest
        Parameters:
            - startDateOfTest: datetime object
                It is the date in which the test of this specimen has started, in reference to the date the experiment started
        """
        #Shift time data
        self.startTimeOfTest=startDateOfTest
        temp = []
        for index, timeValue in enumerate(self.timeData):
            temp.append(timeValue-startDateOfTest)
        #Now exclude all data associated to times before startDateOfTest
        #First, lets find all indices in timeData associated to such times
        deletionIndices = [index for index, value in enumerate(temp) if value < timedelta(days=0)]
        #Now erase them from timeData and displacementData lists
        try:
            self.timeData = self.timeData[deletionIndices[-1]+1:-1]
        except IndexError:
            pass
        for series in self.displacementData:
            try:
                self.displacementData[series] = self.displacementData[series][deletionIndices[-1]+1:-1]
            except IndexError:
                pass
        #Now populate plotTime attribute, as it is useful to plot different experiments started at different times
        try:
            self.plotTime = [value.total_seconds() for value in temp[deletionIndices[-1]+1:-1]]
        except IndexError:
            self.plotTime = [value.total_seconds() for value in temp]
        #Now zero shift all displacement data in relation to the beginning of the test
        for series in self.displacementData:
            zeroShift=self.displacementData[series][0]
            for index, value in enumerate(self.displacementData[series]):
                self.displacementData[series][index] = value - zeroShift

    def unmixChannel(self, displacementSeriesReference, displacementSeriesSecondary):
        """
        Sometime the National system may mix up two consecutive channels, switching the values between them.
        This function allows to unmix the channels, based on the discrepancy of the values in the reference series.
        """
        #This will deal with values that got mixed between the two series
        for index, value in enumerate(displacementSeriesReference):
            if index>0:
                #Check if consecutive values are too far apart in both series
                if abs((value-displacementSeriesReference[index-1])/value)>0.1:
                    #if abs((displacementSeriesSecondary[index]-displacementSeriesSecondary[index-1])/displacementSeriesSecondary[index])>0.1:
                    #Check if consecutive measurements did not occur in two very distincts moments in time
                    #e.g., if the test stopped for a lot of time
                    if self.timeData[index]-self.timeData[index-1]<timedelta(minutes=60):
                        temp = displacementSeriesSecondary[index]
                        displacementSeriesSecondary[index]=displacementSeriesReference[index]
                        displacementSeriesReference[index]=temp
        #Below will deal with values that got repeated on the same channel (the channel reference was read two times, by its own channel and by secondary channel)
        for index, value in enumerate(displacementSeriesSecondary):
            if index>0:
                #Check if the present value is too close to the value from the reference channel
                if abs((value-displacementSeriesReference[index])/value)<0.05:
                    displacementSeriesSecondary[index]=np.NaN
        displacementSeriesSecondary = [x for x in displacementSeriesSecondary if ~np.isnan(x)]
        
    def manualCalibration(self, calibrationCurves):
        """
        Sometimes we may want to input manual calibrations in the code (for National system mainly)
        This function allows to convert raw voltage values to calibrated values in mm.
        Parameters:
            - calibrationCurves: dict
            Dictionary whose keywords contain channel names (str) indexing a list [a,b] in which a is the angular coefficient and b is the intercept of a lienar equation obtianed from calbrating the LVDT
        """
        for series in calibrationCurves:
            a=calibrationCurves[series][0]
            b=calibrationCurves[series][1]
            self.displacementData[series]=[a*value+b for value in self.displacementData[series]]

    def setStartOfCreep(self, startDateCreep):
        '''
        A method to define the instant, in seconds, in which only creep is occurrying, i.e., all load has been applied
        Parameters:
            - startDateCreep: datetime object
                Datetime object that index the point, considering the already offseted series after "setStartOfTest" method,
                in which all load has been applied to the specimen.
                This is done after checking agains loading switch data.
        '''
        self.creepStartTime=(startDateCreep-self.startTimeOfTest).total_seconds()

    def plotLVDTData(self):
        for series in self.displacementData:
            plt.plot(self.timeData, self.displacementData[series], label=series)
        plt.ylabel("Displacement (mm)")
        plt.xlabel("Time (date)")
        plt.title("Displacement data for " + self.specimenID)
        plt.legend()

    def addReleaseCorrection(self, fileName, listOfChannelsToBeCorrected, listOfLinesFromWhichCorrectionIsApplied, correctionToApply):
        '''
        A method to define the points for correcting eventual releases/slips that occur in LVDTs during testing,
        associated to external factors of the experiment, and which create huge jumps in displacement data.
        It populates the attribute self.releaseCorrectionList
        It has to be called once for every file in which release/slip occurs
        Parameters:
            - fileName: str
                A real file name, with .csv, in which the release/slip has occurred
            - listOfChannelsToBeCorrected: list of str
                A list of the names of LVDT channels that will be corrected, according to the names given when constructing this object
            - listOfLinesFromWhichCorrectionIsApplied: list of ints
                A list of the lines in the files (ordered according to the channels to be corrected) beyond which the correction will always be applied
                It has to discount the header (counts from the first line of data)
            - correctionToApply: float
                A correction, in mm, to be applied (summed or subtract) to the experimental data
        '''
        #The line-1 below is justified because Python indexes the 1st line as line 0
        #The last element helps identifying whether this correction was applied to the data or not
        self.releaseCorrectionList[fileName] = [listOfChannelsToBeCorrected, [line-1 for line in listOfLinesFromWhichCorrectionIsApplied], correctionToApply, [False for item in listOfChannelsToBeCorrected]]
                        
class loadCell():
    """
    A class that defines all the parameters of a load cell
    """
    def __init__(self, loadCellID, acquisitionSystem, channels, lineColor, lineStyleForPlot):
        """
        Initialization of class
        Parameters:
            - loadCellID: str
                Defines the identification used to refer to this load cell
            - acquisitionSystem: str
                Defines the acquisition system used to monitor this load cell
            - channels: int
                Defines the column in the text file associated to the load cell used to monitor this specimen.
                To know which numbers identify the load cell, you need to know the position of the load cell channel in the result file
                This assumes that for a single experiment, the test files always follow the same order of channels in the logged data.
            - lineColor: ()
                Defines the line color used to plot the load cell data
            - lineStyleForPlot: ()
                Defines the line style used to plot the load cell data

        Aditional class attributes:
            - loadData: list
                Initiated as an empty list to be populated with load data read from experiment result files
            - timeData: list
                Initiated as an empty list to be populated with time data from experiment result files
            - seriesStartTime: list
                Initiated as an empty list to be populated with the start time, in datetime object, of each series composing the data (data may be composed by several series, each in a different test file) read from experiment result files
        """
        #Attributes defined in class instantiation
        self.specimenID=loadCellID
        self.acquisitionSystem=acquisitionSystem
        self.channels=channels
        self.lineColor=lineColor
        self.lineStyleForPlot=lineStyleForPlot

        #Additional class attributes
        self.loadData=[]
        self.timeData=[]
        self.seriesStartTime=[]

class loadingSwitch():
    """
    A class that defines all the parameters of a loading switch (an Arduino device that allows knowing when the load was completely applied to the specimen)
    """
    def __init__(self):
        """
        Initialization of class
        Parameters:
            - microsecondsData: list of float
                Initiated as an empty list ot be puplated with time data read directly from experiment result files
            - timeData: list of datetime objetcts
                Initiated as an empty list to be populated with time data from experiment result files
            - readValues: list
                Initiated as an empty list to be populated with the values read by the switch
        """
        self.microsecondsData=[]
        self.timeData=[]
        self.voltageData=[]
    
    def addOffset(self, dateTimeOffset, notablePointTimeInLoadingSwitch):
        '''
        A method to time stamp the loadingSwitch data in the same reference as the LVDT creep data.
        Parameters:
            dateTimeOffset: datetime object
                A datetime object that specifies the time of the beginning of this series in terms of the reference of the creep system
                It may be taken as the value of a notable point (peak, valley) of the curves of the pre-test procedure, in which
                a LVDT connected to the creep system is attached to the loading switch and they perform simultanous measurement.
                For praticallity, it may taken as the time registered by the creep system (and posterior offset in terms of 
                the runtime in microseconds of the loading sweitch system may be done with the parameter below)
            notablePointTimeInLoadingSwitch: datetime object
                A datatime object that contains the value of the selected notable point in the loading switch record, so everything can be synchrnozied
        '''
        for instant in self.microsecondsData:
            self.timeData.append(dateTimeOffset+timedelta(microseconds=instant)-notablePointTimeInLoadingSwitch)

    def plotData(self, typeOfTime='raw'):
        if typeOfTime=='raw':
            plt.plot(self.microsecondsData, self.voltageData, label="loading switch")
            plt.xlabel("Time (microseconds)")
        elif typeOfTime=='offset':
            plt.plot(self.timeData, self.voltageData, label="loading switch")
            plt.xlabel("Time (date)")
        plt.ylabel("Digital value (-)")
        plt.legend()
        plt.show()

class experiment():
    """
    A class that defines a creep experiment with multiple test specimens
    """
    def __init__(self, testSpecimensList, loadCellList, loadingSwitchObject=False, numberOfActiveChannels = False):
        #numberOfActiveChannels needs to be set only if not all testSpecimens or loadCells of the experiment
        #are used in the processing of the data
        self.testSpecimensList=testSpecimensList #Receives objects of testSpecimen class
        self.loadCellList=loadCellList #Receives objects of loadCell class
        self.loadingSwitchObject=loadingSwitchObject #Receives objects of loadingSwitch class. If False, not available
        self.numberOfActiveChannels=numberOfActiveChannels
        #Separate specimens in the systems being used
        #This may be expanded as more systems are included among the testing systems
        #The below two variables will keep the indices of testSpecimensList associated to each system (instead of copying everything)
        self.inegiSpecimens=[]
        self.nationalSpecimens=[]
        for index, specimen in enumerate(testSpecimensList):
            if specimen.acquisitionSystem == 'inegi':
                self.inegiSpecimens.append(index)
            elif specimen.acquisitionSystem == "national":
                self.nationalSpecimens.append(index)
            else:
                print("System configured in specimen ",specimen.specimenID," is invalid.")   
                print("Script is being terminated.") 
                exit()
        #Do the same with load cells
        #For now that we only have load cells in INEGI system, this is rather unuseful
        self.inegiLoadCells=[]
        self.nationalLoadCells=[]
        for index, loadCell in enumerate(loadCellList):
            if specimen.acquisitionSystem == 'inegi':
                self.inegiLoadCells.append(index)
            elif specimen.acquisitionSystem == "national":
                self.nationalLoadCells.append(index)
            else:
                print("System configured in load cell ",loadCell.loadCellID," is invalid.")   
                print("Script is being terminated.") 
                exit()

        #Define attributes that will storate statistical measurements of the creep test
        self.interpolatedTimeData=None
        self.averageCompliance=None
        self.interpolatedTimeDataSpecificCreep=None
        self.averageSpecificCreep=None
        self.stdDevCompliance=None
        self.stdDevSpecificCreep=None
        self.coefficientOfVariationCompliance=None
        self.coefficientOfVariationSpecificCreep=None

    def readCreep_Batch(self, path, timeStampingMethod, filterInterval=None):
        """
        This function populates the attributes displacementData, timeData and seriesStartTime of each testSpecimens and load cell object

        Parameters:
            - filterInterval: optional, list of list
                Specifies a list with each component being [start,end], with start and end being datetime objects that specify the start and end times 
                for which data should not be read ofr some reason (ex. some experimental interference that should not appear in the final data)
        """
        #First, lets do it for all INEGI specimens and load cells
        self.readINEGI_Batch(path, timeStampingMethod, filterInterval)
        self.readNational_Batch(path, timeStampingMethod)
        #Now, we read loading switch data
        if self.loadingSwitchObject is not False:
            self.readLoadingSwitch_Batch(path)

    def readINEGI_Batch(self, path, timeStampingMethod, filterInterval=None):
        #Define the number of active channels in the test files
        #For the case all specimens and channels are to be used, then it follows the following logic:
            #This is equal to the number of objects associated to the INEGI system being read
            #In other words, if all data is to be used, it is the number of specimens LVDTs and load cells associated to the INEGI system
        #For the case only a set of the specimens and channels are selected for processing, then it follows another logic:
            #In this case, the algorithm can no longer automatically identifies which channels to read from the file
            #In this case, it is necessary to manually set the parameter numberOfActiveChannels
        numberOfChannels=0
        if self.numberOfActiveChannels is False:
            for index in self.inegiSpecimens:
                numberOfChannels=numberOfChannels+len(self.testSpecimensList[index].channelsLVDT)
            for index in self.inegiLoadCells:
                try:
                    numberOfChannels=numberOfChannels+len(self.loadCellList[index].channels)
                except TypeError:
                    #If TypeError has occurred, it is because this load cell has only one channel, so we add only 1 unit to numberOfChannels
                    numberOfChannels=numberOfChannels+1
        else:
            numberOfChannels=self.numberOfActiveChannels

        #Create empty variables to store values throughout processing
        extension = 'csv'
        testTime=[]
        samplingFrequency=[]
        seriesStartTime=[]
        #Change the current directory to path
        os.chdir(path)
        csvFilesList = glob.glob('*.{}'.format(extension))

        #Each time a new file is started, the series start it all again, because it is relative to the beginning of the test
        #So we have to take into account the reference LVDT values at the start of the experiment
        #We have to instantiate an empty list with the same number of active channels
        #The mapping between the indices of these two lists and the true columns of the result file is simply: true column = indices of list + 1
        #This is because the result file has an extra column, in the beginning, associated to time
        LVDT_reference = [0 for channel in range(0,numberOfChannels)]
        LVDT_correctionFactor = [0 for channel in range(0,numberOfChannels)]
        #Also, sometimes release/slips occur during the test, so we also have to compensate for that
        LVDT_releaseCorrectionFactor = [0 for channel in range(0,numberOfChannels)]

        for iterationFiles,csvFile in enumerate(csvFilesList):
            #Read .csv file
            with open(csvFile) as file:
                csvreader = csv.reader(file,delimiter=';')

                #Extract sampling frequency
                header = []
                header = next(csvreader)
                header = next(csvreader)
                samplingFrequencyInStr=header[1][header[1].index("=")+1:-2].replace(",",".")
                samplingFrequency.append(float(samplingFrequencyInStr))

                #Extract date of beginning of the test
                if timeStampingMethod == 'fileName':
                    header = next(csvreader) #Read this line just so next functions can properly work
                    format_string = "%Y-%m-%d-%H-%M"
                    seriesStartTime.append(datetime.strptime(csvFile[0:16], format_string))    
                elif timeStampingMethod == 'systemAutomatic':
                    header = next(csvreader)
                    fileDateTime=header[1]
                    format_string = "%d-%m-%Y %H:%M:%S"
                    seriesStartTime.append(datetime.strptime(fileDateTime, format_string))
                else:
                    print("Inexistent time stamping method for this system.")
                    print("Script is being terminated.")
                    exit()

                #Read rest of header data to start at the first row of referece data
                for i in range(0,5):
                    header = next(csvreader)

                #Read reference data
                #The first reference data line is different, it begins with "Transducer"
                if iterationFiles == 0: 
                    #It is the first file, of the beginning of the test, so lets store our reference
                    for i in range(0,numberOfChannels):
                        header = next(csvreader)
                        referenceInStr=header[2].replace(",",".")
                        LVDT_reference[i]=float(referenceInStr)
                else:
                    for i in range(0,numberOfChannels):
                        #It is not the first file, so let's compute the correction factor
                        header = next(csvreader)
                        referenceInStr=header[2].replace(",",".")
                        LVDT_correctionFactor[i]=float(referenceInStr)-LVDT_reference[i]

                '''
                #If load cells are present, read their lines
                #TODO: Implement a storage of these values so we can plot load cell evolution
                for i in range(0,numberOfLoadCells):
                    header = next(csvreader)
                '''

                #Read rest of header data to start at the first row of measurement data
                for i in range(0,4):
                    header = next(csvreader)

                for iterationRows, row in enumerate(csvreader):
                    ## Read time data
                    rawTime=(row[0])
                    #FIRST, WE USED THE FILES TIME STAMPT TO TIME STAMP THE EXPERIMENTAL MEASUREMENTS
                    #HOWEVER, IT WAS OBSERVED FROM SYNCHRONICITY COMPARISONS TO THE CREEP SWITCH DEVICE THAT THOSE STAMPS WERE UNRELIABLE
                    #AND THAT THE SYSTEM WAS ACTUALLY MEASURING CONSISTENTLY WITH THE CONFIGURED SAMPLING FREQUENCY
                    #THAT'S WHY WE CHANGED THE WAY WE TIME STAMP THE SERIES BY USING ONLY THE SAMPLING FREQUENCY
                    #Create format string for parsing string to datetime object
                    '''
                    format_string = "%d:%H:%M:%S.%f"
                    processedTime = rawTime[:-6] + "." + rawTime[-5:-2] + rawTime[-1:]
                    '''
                    if len(rawTime)==14:
                        #This will have to be done if time read value has 14 characters
                        #This means that a day haven't passed yet (we are in the first 24 hours of the test)
                        #For the function strptime does not work, we have to artificially add 01 day to the time data (it doesnt accept a day = 00)
                        #When we finally compute samplingTime accounting for timedelta, in the last line of this if-clause, we don't consider this artificially added day
                        '''
                        processedTime = "01:" + processedTime
                        samplingTime=datetime.strptime(processedTime, format_string)
                        samplingTime=seriesStartTime[iterationFiles]+timedelta(days=0,hours=samplingTime.hour, minutes=samplingTime.minute, seconds=samplingTime.second, microseconds=samplingTime.microsecond)
                        '''
                        samplingTime=seriesStartTime[iterationFiles]+timedelta(seconds=iterationRows*(1/samplingFrequency[iterationFiles]))
                    else:
                        #If rawTime don't have 14 characters, it will have more and so it will have a day already, so we don't have to add it
                        #Here the system might suffer from overflow of the internal clock (test running for more than 3 days), so we need to also handle it
                        '''
                        try:
                            samplingTime=datetime.strptime(processedTime, format_string)
                            samplingTime=seriesStartTime[iterationFiles]+timedelta(days=samplingTime.day,hours=samplingTime.hour, minutes=samplingTime.minute, seconds=samplingTime.second, microseconds=samplingTime.microsecond)
                        except ValueError:
                            #Overflow has occurred, so we nee to rely on sampling frequency to keep up with data time series
                            samplingTime=testTime[-1]+timedelta(seconds=(1/samplingFrequency[iterationFiles]))
                        '''
                        samplingTime=seriesStartTime[iterationFiles]+timedelta(seconds=iterationRows*(1/samplingFrequency[iterationFiles]))
                    
                    #Check if this datapoint is within ignoring intervals and handle the situation properrly
                    skipReading=False
                    if filterInterval is not None:
                        for timeInterval in filterInterval:
                            if timeInterval[0] < samplingTime < timeInterval[1]:
                                skipReading=True
                                break
                    if skipReading is True:
                        continue #Skip this reading 
                    else: 
                        testTime.append(samplingTime)

                    #Read measured data
                    #Also convert from decimal comma to point
                    #Read LVDT data
                    for specimenIndex in self.inegiSpecimens:
                        currentSpecimen = self.testSpecimensList[specimenIndex]
                        currentSpecimen.timeData.append(samplingTime)
                        
                        #Check if there is need for any correction regarding LVDT release/slip occurred in this test file
                        if csvFile in currentSpecimen.releaseCorrectionList:
                            releaseCorrectionInfo = currentSpecimen.releaseCorrectionList[csvFile]
                            #Inside releaseCorrectionInfo, there will be, ex.: ["SP2-LVDT_5"],[572],[0.16647228],[False]
                            for indexRelease, releasedChannel in enumerate(releaseCorrectionInfo[0]):
                                if currentSpecimen.releaseCorrectionList[csvFile][3][indexRelease] is False:
                                    if iterationRows >= releaseCorrectionInfo[1][indexRelease]:
                                        #If current row is equal or past the line for which the slip has occurred, 
                                        #for this particular channel, proceed to add this correction
                                        channelToBeCorrectedIndex=currentSpecimen.channelsName.index(releasedChannel)
                                        columnIndexInTestFile=currentSpecimen.channelsLVDT[channelToBeCorrectedIndex]
                                        #Update LVDT_releaseCorrectionFactor
                                        LVDT_releaseCorrectionFactor[columnIndexInTestFile-1]=LVDT_releaseCorrectionFactor[columnIndexInTestFile-1]+releaseCorrectionInfo[2][indexRelease]
                                        #Update currentSpecimen.releaseCorrectionList to signal correction was already applied to LVDT_releaseCorrectionFactor
                                        currentSpecimen.releaseCorrectionList[csvFile][3][indexRelease]=True
                        
                        #Finally read LVDT data
                        for index, channel in enumerate(currentSpecimen.channelsLVDT):
                            currentChannelName=currentSpecimen.channelsName[index]
                            value = row[channel].replace(",",".")
                            currentSpecimen.displacementData[currentChannelName].append(float(value)+LVDT_correctionFactor[channel-1]+LVDT_releaseCorrectionFactor[channel-1]) #Check mapping of LVDT_correctionFactor variable in its definition above
                        
                    #Read load cell data
                    for loadCellIndex in self.inegiLoadCells:
                        currentLoadCell = self.loadCellList[loadCellIndex]
                        currentLoadCell.timeData.append(samplingTime)
                        value = row[currentLoadCell.channels].replace(",",".")
                        currentLoadCell.loadData.append(float(value)+LVDT_correctionFactor[currentLoadCell.channels-1]) #Check mapping of LVDT_correctionFactor variable in its definition above
                    
            #Store seriesStartTime variable
            for specimenIndex in self.inegiSpecimens:
                currentSpecimen = self.testSpecimensList[specimenIndex]
                currentSpecimen.seriesStartTime=seriesStartTime 

    def readNational_Batch(self, path, timeStampingMethod):
        #Define the number of active channels in the test files
        #This is equal to the number of objects associated to the INEGI system being read
        #In other words, it is the number of specimens LVDTs and load cells associated to the INEGI system
        numberOfChannels=0
        for index in self.nationalSpecimens:
            numberOfChannels=numberOfChannels+len(self.testSpecimensList[index].channelsLVDT)
        for index in self.nationalLoadCells:
            try:
                numberOfChannels=numberOfChannels+len(self.loadCellList[index].channels)
            except TypeError:
                #If TypeError has occurred, it is because this load cell has only one channel, so we add only 1 unit to numberOfChannels
                numberOfChannels=numberOfChannels+1

        #Create empty variables to store values throughout processing
        extension = 'txt'
        seriesStartTime=[]
        #Change the current directory to path
        os.chdir(path)
        txtFilesList = glob.glob('*.{}'.format(extension))

        #Each time a new file is started, the series start it all again, because it is relative to the beginning of the test
        #So we have to take into account the reference LVDT values at the start of the experiment
        #We have to instantiate an empty list with the same number of active channels
        #The mapping between the indices of these two lists and the true columns of the result file is simply: true column = indices of list + 1
        #This is because the result file has an extra column, in the beginning, associated to time
        LVDT_reference = [0 for channel in range(0,numberOfChannels)]

        for iterationFiles,csvFile in enumerate(txtFilesList):
            #Read .csv file
            with open(csvFile) as file:
                csvreader = csv.reader(file,delimiter='\t')

                #Extract sampling frequency
                header = []

                #Extract date of beginning of the test
                if timeStampingMethod == 'fileName':
                    format_string = "%Y-%m-%d_%H-%M-%S"
                    seriesStartTime.append(datetime.strptime(csvFile[-23:-4], format_string))    
                else:
                    print("Inexistent time stamping method for this system.")
                    print("Script is being terminated.")
                    exit()

                #Read rest of header data to start at the first row of referece data
                for i in range(0,5):
                    header = next(csvreader)
                
                for iterationRows, row in enumerate(csvreader):
                    ## Read time data
                    rawTime=(row[0])
                    #Create format string for parsing string to datetime object
                    format_string = "%Y/%m/%d %H:%M:%S"
                    #Extract time data from value read from file and append to time array
                    samplingTime=datetime.strptime(rawTime, format_string)
                    
                    #Build map to allow handling need for uncalibrated data
                    adjustForColumnPositionReference=[0 for specimen in self.testSpecimensList]
                    adjustForColumnPositionData=[0 for specimen in self.testSpecimensList]
                    for specimenIndex in self.nationalSpecimens:
                        currentSpecimen = self.testSpecimensList[specimenIndex]
                        if currentSpecimen.uncalibratedValues == True:
                            adjustForColumnPositionReference[specimenIndex]=2
                            adjustForColumnPositionData[specimenIndex]=10
                        else:
                            adjustForColumnPositionReference[specimenIndex]=12
                            adjustForColumnPositionData[specimenIndex]=0
                    
                    #Read reference data if it is the first line of the first file
                    if (iterationRows==0) and (iterationFiles == 0): 
                        #The reference is the first line of the first file
                        for specimenIndex in self.nationalSpecimens:
                            currentSpecimen = self.testSpecimensList[specimenIndex]
                            for index, channel in enumerate(currentSpecimen.channelsLVDT):
                                value = row[channel-12+adjustForColumnPositionReference[specimenIndex]]
                                LVDT_reference[channel-12]=float(value)
                                if currentSpecimen.absoluteValues==True:
                                    LVDT_reference[channel-12]=0
                                    
                    #Read measured data
                    for specimenIndex in self.nationalSpecimens:
                        currentSpecimen = self.testSpecimensList[specimenIndex]
                        currentSpecimen.timeData.append(samplingTime)
                        #Now, store displacement data
                        for index, channel in enumerate(currentSpecimen.channelsLVDT):
                            currentChannelName=currentSpecimen.channelsName[index]
                            value = row[channel-adjustForColumnPositionData[specimenIndex]]
                            currentSpecimen.displacementData[currentChannelName].append(float(value)-LVDT_reference[channel-12]) #Check mapping of LVDT_correctionFactor variable in its definition above                            
            #Store seriesStartTime variable
            for specimenIndex in self.nationalSpecimens:
                currentSpecimen = self.testSpecimensList[specimenIndex]
                currentSpecimen.seriesStartTime=seriesStartTime 

    def readLoadingSwitch_Batch(self, path):
        '''
        This method reads the data from the loading switch system.
        The data must be in the custom format ".LSA" (to do that, just change the file extension from .txt to .lsa manually)
        '''
        #Create empty variables to store values throughout processing
        extension = 'lsa'
        seriesStartTime=[]
        #Change the current directory to path
        os.chdir(path)
        lsaFilesList = glob.glob('*.{}'.format(extension))

        for iterationFiles,csvFile in enumerate(lsaFilesList):
            #Read .lsa file as csv file
            with open(csvFile) as file:
                csvreader = csv.reader(file)
                for iterationRows, row in enumerate(csvreader):
                    ## Read time data
                    self.loadingSwitchObject.microsecondsData.append(int(row[1]))
                    self.loadingSwitchObject.voltageData.append(-(int(row[0])-1023))

    def computeStrainHistory(self):
        for specimen in self.testSpecimensList:
            for series in specimen.displacementData:
                for value in specimen.displacementData[series]:
                    specimen.strainData[series].append((value/(specimen.baseLength)))

    def computeCompliances(self):
        for specimen in self.testSpecimensList:
            for series in specimen.displacementData:
                for value in specimen.displacementData[series]:
                    specimen.complianceData[series].append((value/(specimen.baseLength))/specimen.appliedStress)

    def computeSpecificCreep(self):
        """
        This function needs to be used after using computeCompliances(), since it will use the attribute specimen.complianceData to compute specific creep.
        It also requires defining the specimen's attribute self.creepStartTime, otherwise it will throw an error.
        """
        shiftTime=0
        for specimen in self.testSpecimensList:
            for indexSeries, series in enumerate(specimen.complianceData):
                creepStarted=False
                for index, value in enumerate(specimen.complianceData[series]):
                    if specimen.plotTime[index]>specimen.creepStartTime:
                        if creepStarted==False:
                            shiftTime=specimen.plotTime[index]
                            shiftCompliance=specimen.complianceData[series][index]
                            creepStarted=True
                        if indexSeries==0:
                            #Only build the time series one time, and not three times for each LVDT line
                            specimen.specificCreepTimeData.append(specimen.plotTime[index]-shiftTime)
                        specimen.specificCreep[series].append(specimen.complianceData[series][index]-shiftCompliance)
                    else:
                        #Creep hasnt started, do nothing
                        pass

    def computeStatisticalMeasures(self):
        """
        This method computes the average compliance and the associated standard deviation and coefficient of variation fof the data set
        It needs to be used after computeSpecificCreep()
        It populates the attributes:
            -self.interpolatedTimeData=None
            -self.averageCompliance=None
            -self.stdDevCompliance=None
            -self.coefficientOfVariationCompliance=None
            -self.averageSpecificCreep=None
        """
        import numpy as np

        #Gather data sets from each specimen in the experiment
        complianceData=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        specificCreepData=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        timeData=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        specificCreepTimeData=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        for iteration, specimen in enumerate(self.testSpecimensList):
            for series in specimen.displacementData:
                complianceData[iteration] = [valuey+valuedict for valuey,valuedict in zip(complianceData[iteration],specimen.complianceData[series])]
                specificCreepData[iteration] = [valuey+valuedict for valuey,valuedict in zip(specificCreepData[iteration],specimen.specificCreep[series])]
            complianceData[iteration]=[1000000*value/len(specimen.displacementData) for value in complianceData[iteration]]
            specificCreepData[iteration]=[1000000*value/len(specimen.displacementData) for value in specificCreepData[iteration]]
            timeData[iteration]=specimen.plotTime
            specificCreepTimeData[iteration]=specimen.specificCreepTimeData
        
        #Interpolate each data set from each specimen accordingly to the time interval available for averaging
        minimumTime=max([min(value) for value in timeData])
        maximumTime=min([max(value) for value in timeData])
        minimumTimeSpecificCreep=max([min(value) for value in specificCreepTimeData])
        maximumTimeSpecificCreep=min([max(value) for value in specificCreepTimeData])
        #Create new time data vector for interpolate
        self.interpolatedTimeData=np.arange(minimumTime, maximumTime, 1)
        self.interpolatedTimeDataSpecificCreep=np.arange(minimumTimeSpecificCreep, maximumTimeSpecificCreep, 1)
        #Interpolate compliance data
        complianceDataInterp=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        for iteration, specimen in enumerate(self.testSpecimensList):
            complianceDataInterp[iteration]=  np.interp(self.interpolatedTimeData, timeData[iteration], complianceData[iteration])
        #Interpolate specific compliance data
        specificCreepDataInterp=[[0 for values in specimen.displacementData[list(specimen.displacementData.keys())[0]]] for specimen in self.testSpecimensList]
        for iteration, specimen in enumerate(self.testSpecimensList):
            specificCreepDataInterp[iteration]=  np.interp(self.interpolatedTimeDataSpecificCreep, specificCreepTimeData[iteration], specificCreepData[iteration])
            
        #Compute statistical measurements of the dataset
        self.averageCompliance = np.mean(np.array(complianceDataInterp), axis=0)
        self.stdDevCompliance = np.std(np.array(complianceDataInterp), axis=0)
        self.coefficientOfVariationCompliance = np.divide(self.stdDevCompliance,self.averageCompliance)
        self.averageSpecificCreep = np.mean(np.array(specificCreepDataInterp), axis=0)
        self.stdDevSpecificCreep = np.std(np.array(specificCreepDataInterp), axis=0)
        self.coefficientOfVariationSpecificCreep = np.divide(self.stdDevSpecificCreep,self.averageSpecificCreep)

    #Methods for visualization of results
    def pltDisplacementData(self, logScale = False, normalized = False):
        """
        This method plots the displacement data, in micrometers.
        """
        import numpy as np
        for specimen in self.testSpecimensList:
            listLVDT = list(specimen.displacementData.keys())
            y=[0 for values in specimen.displacementData[listLVDT[0]]]
            for series in specimen.displacementData:
                y = [valuey+valuedict for valuey,valuedict in zip(y,specimen.displacementData[series])]
            if normalized is False:
                plt.plot(np.array(specimen.plotTime)/(60*60*24), [1000*value/len(specimen.displacementData) for value in y], label=specimen.specimenID)
            else:
                y_plot=[1000*value/(len(specimen.displacementData)*normalizedValue) for value in y]
                normalizedValue = max(y_plot)
                y_plot=[value/normalizedValue for value in y_plot]
                plt.plot(np.array(specimen.plotTime)/(60*60*24), y_plot, label=specimen.specimenID)
        
        if normalized is False:
            plt.ylabel(r'Displacement [µm]')
        else:
            plt.ylabel(r'Normalized displacement [-]')
        plt.xlabel("Time [days]")
        plt.legend()

        if logScale is True:
            plt.xscale("log")

        plt.show()
    
    def pltSpecimenSpecificCreep(self, logScale = False, normalized = False):
        """
        This method plots the specific creep of each specimen of the experiment, in [µε/MPa]
        It needs to be used after using computeCompliances(), since it will use the attribute specimen.complianceData to plot the data.
        """
        import numpy as np
        for specimen in self.testSpecimensList:
            listLVDT = list(specimen.specificCreep.keys())
            y=[0 for values in specimen.specificCreep[listLVDT[0]]]
            for series in specimen.specificCreep:
                y = [valuey+valuedict for valuey,valuedict in zip(y,specimen.specificCreep[series])]
            if normalized is False:
                plt.plot([value/(60*60*24) for value in specimen.specificCreepTimeData], [value/3 for value in y], label=specimen.specimenID)
            else:
                y_plot= specimen.specificCreep
                normalizedValue = max(y_plot)
                y_plot=[value/normalizedValue for value in y_plot]
                plt.plot([value/(60*60*24) for value in specimen.specificCreepTimeData], y_plot, label=specimen.specimenID)
        
        if normalized is False:
            plt.ylabel(r'Specific creep [µε/MPa]')
        else:
            plt.ylabel(r'Normalized specific creep [-]')
        plt.xlabel("Time [days]")
        plt.legend()

        if logScale is True:
            plt.xscale("log")

        plt.show()

    def pltAverageCompliance(self, logScale = False, normalized = False, stdInterval=False, specimenResults=False, title=None):
        """
        This method plots the average compliance considering all the specimens of the experiment, in [µε/MPa]
        It needs to be used after using computeStatisticalMeasures()
        """
        if normalized is False:
            plotAverageCompliance=self.averageCompliance
            plotStdDevCompliance=self.stdDevCompliance
        else:
            plotAverageCompliance=self.averageCompliance/max(self.averageCompliance)
            plotStdDevCompliance=self.stdDevCompliance/max(self.averageCompliance)

        if stdInterval is True:
            plt.fill_between(self.interpolatedTimeData/(60*60*24), plotAverageCompliance+plotStdDevCompliance, plotAverageCompliance-plotStdDevCompliance, color=(0.9, 0.9, 0.9), label=None)
            plt.plot(self.interpolatedTimeData/(60*60*24), plotAverageCompliance+plotStdDevCompliance, "-", color="grey", label=None, linewidth=1)
            plt.plot(self.interpolatedTimeData/(60*60*24), plotAverageCompliance-plotStdDevCompliance, "-", color="grey", label=None, linewidth=1)
        
        if specimenResults is True:
            for specimen in self.testSpecimensList:
                listLVDT = list(specimen.complianceData.keys())
                y=[0 for values in specimen.complianceData[listLVDT[0]]]
                for series in specimen.complianceData:
                    y = [valuey+valuedict for valuey,valuedict in zip(y,specimen.complianceData[series])]
                if normalized is False:
                    plt.plot([value/(60*60*24) for value in specimen.plotTime], [(1e6)*value/3 for value in y], label=specimen.specimenID)
                else:
                    y_plot= specimen.specificCreep
                    normalizedValue = max(y_plot)
                    y_plot=[value/normalizedValue for value in y_plot]
                    plt.plot([value/(60*60*24) for value in specimen.plotTime], y_plot, label=specimen.specimenID)

        plt.plot(self.interpolatedTimeData/(60*60*24), plotAverageCompliance, color="black", label="Average")

        if normalized is False:
            plt.ylabel(r'Compliance [µε/MPa]')
        else:
            plt.ylabel(r'Normalized compliance [-]')
        plt.xlabel("Time [days]")
        plt.legend()

        if logScale is True:
            plt.xscale("log")

        plt.show()
    
    def pltAverageSpecificCreep(self, logScale = False, normalized = False, stdInterval=False, specimenResults=False, title=None):
        """
        This method plots the average specific creep considering all the specimens of the experiment, in [µε/MPa]
        It needs to be used after using computeStatisticalMeasures()
        """
        if normalized is False:
            plotAverageSpecificCreep=self.averageSpecificCreep
            plotStdDevSpecificCreep=self.stdDevSpecificCreep
        else:
            plotAverageSpecificCreep=self.averageSpecificCreep/max(self.averageSpecificCreep)
            plotStdDevSpecificCreep=self.stdDevSpecificCreep/max(self.averageSpecificCreep)

        if stdInterval is True:
            plt.fill_between(self.interpolatedTimeDataSpecificCreep/(60*60*24), plotAverageSpecificCreep+plotStdDevSpecificCreep, plotAverageSpecificCreep-plotStdDevSpecificCreep, color=(0.9, 0.9, 0.9), label=None)
            plt.plot(self.interpolatedTimeDataSpecificCreep/(60*60*24), plotAverageSpecificCreep+plotStdDevSpecificCreep, "-", color="grey", label=None, linewidth=1)
            plt.plot(self.interpolatedTimeDataSpecificCreep/(60*60*24), plotAverageSpecificCreep-plotStdDevSpecificCreep, "-", color="grey", label=None, linewidth=1)
        
        if specimenResults is True:
            for specimen in self.testSpecimensList:
                listLVDT = list(specimen.specificCreep.keys())
                y=[0 for values in specimen.specificCreep[listLVDT[0]]]
                for series in specimen.specificCreep:
                    y = [valuey+valuedict for valuey,valuedict in zip(y,specimen.specificCreep[series])]
                if normalized is False:
                    plt.plot([value/(60*60*24) for value in specimen.specificCreepTimeData], [(1e6)*value/3 for value in y], label=specimen.specimenID)
                else:
                    y_plot= specimen.specificCreep
                    normalizedValue = max(y_plot)
                    y_plot=[value/normalizedValue for value in y_plot]
                    plt.plot([value/(60*60*24) for value in specimen.specificCreepTimeData], y_plot, label=specimen.specimenID)
        
        plt.plot(self.interpolatedTimeDataSpecificCreep/(60*60*24), plotAverageSpecificCreep, color="black", label="Average")


        if normalized is False:
            plt.ylabel(r'Specific creep [µε/MPa]')
        else:
            plt.ylabel(r'Normalized specific creep [-]')
        plt.xlabel("Time [days]")
        plt.legend()

        if logScale is True:
            plt.xscale("log")

        if title is not None:
            plt.title(title)

        plt.show()

    def pltCoefficientOfVariationSpecificCreep(self, logScale = False):
        plt.plot(self.interpolatedTimeData/(60*60*24), 100*self.coefficientOfVariationCompliance, color="black", label="Average")
        plt.xlabel("Time [days]")
        plt.ylabel("Coefficient of variation (%)")
        plt.legend()
        if logScale is True:
            plt.xscale("log")
        plt.show()

    #Methods for saving results in csv files
    def saveAverageDisplacementData(self, fileName):
        """
        This method plots the displacement data, in micrometers.
        """
        import pandas as pd
        import numpy as np

        # Creating Empty DataFrame and Storing it in variable df
        df = pd.DataFrame()

        for specimen in self.testSpecimensList:
            #Get data from the specimen
            listLVDT = list(specimen.displacementData.keys())
            y=[0 for values in specimen.displacementData[listLVDT[0]]]
            for series in specimen.displacementData:
                y = [valuey+valuedict for valuey,valuedict in zip(y,specimen.displacementData[series])]
            #Build tuple with time + displacement data
            list_of_tuples = list(zip(np.array(specimen.plotTime), [1000*value/len(specimen.displacementData) for value in y]))
            # Converting lists of tuples into pandas Dataframe.
            temporaryDataframe = pd.DataFrame(list_of_tuples, columns=[f'{specimen.specimenID}-time', f'{specimen.specimenID}-averageDisplacement'])
            df = pd.concat([df, temporaryDataframe], axis=1)
        df.to_csv(f'{fileName}.csv',index=False)

    def saveAverageCompliance(self, fileName):
        """
        This method plots the displacement data, in micrometers.
        """
        import pandas as pd
        import numpy as np

        list_of_tuples = list(zip(self.interpolatedTimeData, self.averageCompliance, self.stdDevCompliance))
        df = pd.DataFrame(list_of_tuples, columns=["interpolatedTime", "averageCompliance", "stdDevCompliance"])

        df.to_csv(f'{fileName}.csv',index=False)
    
    def saveAverageSpecificCreep(self, fileName):
        """
        This method plots the displacement data, in micrometers.
        """
        import pandas as pd
        import numpy as np

        list_of_tuples = list(zip(self.interpolatedTimeData, self.averageSpecificCreep, self.stdDevSpecificCreep))
        df = pd.DataFrame(list_of_tuples, columns=["interpolatedTime", "averageCompliance", "stdDevCompliance"])

        df.to_csv(f'{fileName}.csv',index=False)

    ## Functions to pickle and store an experiment object
    def saveExperimentObject(self, fileName):
        output = open(fileName, 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(self, output)
        output.close()

#Old functions from the "functional" implementation of this module

def readCreepCSV_Batch(path, timeStampingMethod, numberOfChannels, numberOfLoadCells=0):
    """
    Read all creep test csv files inside path.
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    path: str
        Complete path of the file
    timeStampingMethod: str
        Can be 'fileName' (get time data from the name of the data file) or 'systemAutomatic' (get time data from the value stored by the system).
    numberOfChannels: int
        Informs the number of active channels in the csv files
    numberOfLoadCells: int
        Informes the number of active load cells. They will be always the last two columns

    Returns
    -------    
    LVDT: list
        List containing the displacement values measured in the creep test, for each of the channels.
    testTime: list
        List containing the time stamp of each displacement value
    seriesStartTime: list
        List containing the start time, in datetime object, of each series composing the data (data may be composed by several series, each in a different test file)
    """ 

    #Create empty variables to store values throughout processing
    extension = 'csv'
    testTime=[]
    LVDT=[[] for channel in range(0,numberOfChannels)] #LVDT array that will store [[LVDT_1],[LVDT_3],[LVDT_5],[LVDT_2],[LVDT_4],[LVDT_6]]
    samplingFrequency=[]
    seriesStartTime=[]

    #Change the current directory to path
    os.chdir(path)
    csvFilesList = glob.glob('*.{}'.format(extension))

    #Each time a new file is started, the series start it all again, because it is relative to the beginning of the test
    #So we have to take into account the reference LVDT values at the start of the experiment
    #We have to put here as many "zeroes" as the number of active channels
    LVDT_reference = [0 for channel in range(0,numberOfChannels)]
    LVDT_correctionFactor = [0 for channel in range(0,numberOfChannels)]

    for iterationFiles,csvFile in enumerate(csvFilesList):
        #Read .csv file
        with open(csvFile) as file:
            csvreader = csv.reader(file,delimiter=';')

            #Extract sampling frequency
            header = []
            header = next(csvreader)
            header = next(csvreader)
            samplingFrequencyInStr=header[1][header[1].index("=")+1:-2].replace(",",".")
            samplingFrequency.append(float(samplingFrequencyInStr))

            #Extract date of beginning of the test
            if timeStampingMethod == 'fileName':
                header = next(csvreader) #Read this line just so next functions can properly work
                format_string = "%Y-%m-%d-%H-%M"
                seriesStartTime.append(datetime.strptime(csvFile[0:16], format_string))    
            elif timeStampingMethod == 'systemAutomatic':
                header = next(csvreader)
                fileDateTime=header[1]
                format_string = "%d-%m-%Y %H:%M:%S"
                seriesStartTime.append(datetime.strptime(fileDateTime, format_string))
            else:
                print("Inexistent time stamping method!")

            #Read rest of header data to start at the first row of referece data
            for i in range(0,5):
                header = next(csvreader)

            #Read reference data
            #The first reference data line is different, it begins with "Transducer"
            if iterationFiles == 0: 
                #It is the first file, of the beginning of the test, so lets store our reference
                for i in range(0,numberOfChannels):
                    header = next(csvreader)
                    referenceInStr=header[2].replace(",",".")
                    LVDT_reference[i]=float(referenceInStr)
            else:
                for i in range(0,numberOfChannels):
                    #It is not the first file, so let's compute the correction factor
                    header = next(csvreader)
                    referenceInStr=header[2].replace(",",".")
                    LVDT_correctionFactor[i]=float(referenceInStr)-LVDT_reference[i]

            #If load cells are present, read their lines
            #TODO: Implement a storage of these values so we can plot load cell evolution
            for i in range(0,numberOfLoadCells):
                header = next(csvreader)

            #Read rest of header data to start at the first row of measurement data
            for i in range(0,4):
                header = next(csvreader)

            for iterationRows, row in enumerate(csvreader):
                ## Read time data
                rawTime=(row[0])
                #Create format string for parsing string to datetime object
                format_string = "%d:%H:%M:%S.%f"
                #Extract time data from value read from file. The decimal place in the milisseconds is ignored for simplicity of code
                processedTime = rawTime[:-6] + "." + rawTime[-5:-2]
                if len(rawTime)==14:
                    #This will have to be done if time read value has 14 characters
                    #This means that a day haven't passed yet (we are in the first 24 hours of the test)
                    #For the function strptime does not work, we have to artificially add 01 day to the time data (it doesnt accept a day = 00)
                    #When we finally compute samplingTime accounting for timedelta, in the last line of this if-clause, we don't consider this artificially added day
                    processedTime = "01:" + processedTime
                    samplingTime=datetime.strptime(processedTime, format_string)
                    samplingTime=seriesStartTime[iterationFiles]+timedelta(days=0,hours=samplingTime.hour, minutes=samplingTime.minute, seconds=samplingTime.second, microseconds=samplingTime.microsecond)
                else:
                    #If rawTime don't have 14 characters, it will have more and so it will have a day already, so we don't have to add it
                    #Here the system might suffer from overflow of the internal clock (test running for more than 3 days), so we need to also handle it
                    try:
                        samplingTime=datetime.strptime(processedTime, format_string)
                        samplingTime=seriesStartTime[iterationFiles]+timedelta(days=samplingTime.day,hours=samplingTime.hour, minutes=samplingTime.minute, seconds=samplingTime.second, microseconds=samplingTime.microsecond)
                    except ValueError:
                        #Overflow has occurred, so we nee to rely on sampling frequency to keep up with data time series
                        samplingTime=testTime[-1]+timedelta(seconds=(1/samplingFrequency[iterationFiles]))
                #Append time to the array 
                testTime.append(samplingTime)

                ## Read LVDT data
                #First convert from decimal comma to point
                for lvdtChannel, value in enumerate(row[1:-1-numberOfLoadCells]):
                    value = value.replace(",", ".")
                    LVDT[lvdtChannel].append(float(value)+LVDT_correctionFactor[lvdtChannel])
    return LVDT, testTime, seriesStartTime          

def convertToMicrometers(LVDT):
    """
    Convert LVDT data initially in mm to micrometers
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    LVDT: list
        List containing the displacement values measured in the creep test, for each of the 

    Returns
    -------    
    convertedLVDT: list
        Values converted to micrometers
    """ 
    convertedLVDT = [[] for lvdtChannel in LVDT]
    for indexChannel, lvdtChannel in enumerate(LVDT):
        for indexValue, value in enumerate(lvdtChannel):
            convertedLVDT[indexChannel].append(1000*LVDT[indexChannel][indexValue])
    return convertedLVDT

def convertDeltaTime(testTime, startTime, desiredUnit):
    """
    Convert raw experimental time data to delta times to a given reference in a desidered unit (seconds, minutes, hours, days)
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.
    
    
    Parameters
    -------       
    testTime: list
        List containing the time stamp of each displacement value
    startTime: datetime object
        Reference time to which delta times willl be computed
    desiredUnit: str
        Can be "minutes", "hours", "days". Specify which conversion will be made

    Returns
    -------    
    convertedTime: list
        Values converted time
    """ 
    #
    convertedTime = []
    for times in testTime:
        delta=times-startTime
        if desiredUnit == "days":
            convertedTime.append(delta.total_seconds()/(60*60*24))
        elif desiredUnit == "hours":
            convertedTime.append(delta.total_seconds()/(60*60))
        elif desiredUnit == "minutes":
            convertedTime.append(delta.total_seconds()/(60))
        else:
            print("Error! Conversion not found")
    
    return convertedTime
    
def averageLVDT(LVDT,channelsToAverage):
    """
    Convert LVDT data initially in mm to micrometers
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    LVDT: list
        List containing the displacement values measured in the creep test, for each of the 
    channelsToAverage: list
        List with the channels that should be considered during averaging, from "1" to "n". For example, [1,2,3] averages the first three channels of the datafiles
    Returns
    -------    
    averagedLVDT: list
        Averaged LVDT values
    """ 
    averagedLVDT = [] 
    tempVariable=0
    for indexValue, value in enumerate(LVDT[0]):
        for channel in channelsToAverage:
            tempVariable=tempVariable+LVDT[channel-1][indexValue]
        averagedLVDT.append(tempVariable/len(channelsToAverage))
        tempVariable=0 #Restart this variable for next iteration
    return averagedLVDT

def averagedCompliance(LVDT, channelsToAverage, baseLength, appliedStress, unitDesired):
    """
    Compute the J from the measured displacements.
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    LVDT: list
        List containing the all the displacement values obtained in the test
    channelsToAverage: list
        List with the channels that should be considered during averaging, from "1" to "n". For example, [1,2,3] averages the first three channels of the datafiles
    baseLength: float
        Base length of the LVDTs, in mm
    appliedStress: float
        Applied stress in MPa to the specimen
    unitDesired: str
        Can be: '1/GPa', 'microns/MPa', 'normalized'
    Returns
    -------    
    averagedValues: list
        Averaged compliance values
    """ 
    
    averagedValues = [] 
    tempVariable=0

    if unitDesired == '1/GPa':
        appliedStress=appliedStress/1000
    elif unitDesired == 'microns/MPa':
        baseLength=baseLength/1000000

    for indexValue, value in enumerate(LVDT[0]):
        try:
            for channel in channelsToAverage:
                tempVariable=tempVariable+LVDT[channel-1][indexValue]
            averagedValues.append(tempVariable/(len(channelsToAverage)*baseLength*appliedStress))
            tempVariable=0 #Restart this variable for next iteration
        except ZeroDivisionError:
            averagedValues.append(0)
    
    if unitDesired == 'normalized':
        lastValue=averagedValues[-1]
        averagedValues[:]=[value/lastValue for value in averagedValues]
        
    return averagedValues

def removeInitialDisplacementWithTime(compliance, timeSeries, initialDispInstant):
    """
    Remove the initial displacement from the compliance series, based on a given time instant from timeSeries.
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    compliance: list
        List containing the all the compliance values
    timeSeries: list
        List containing the all the time stamps of the compliance values
    inititalDispInstant: float
        Instant associataed to the initial displacement to be removed. Same units of timeSeries

    Returns
    -------    
    creepCompliance: list
        List containing all the compliance values with initial displacement removed
    creepComplianceTimeSeries: list
        List containing the time stamps of the creepCompliance list
    """ 
    
    initialDisplacementIndex=next(idx for idx, value in enumerate(timeSeries) if value > initialDispInstant)

    creepCompliance=[(value-compliance[initialDisplacementIndex]) for value in compliance[initialDisplacementIndex:-1]]
    creepComplianceTimeSeries=timeSeries[initialDisplacementIndex:-1]
        
    return creepCompliance, creepComplianceTimeSeries

def downsampleTimeDomain(compliance, timeSeries, timeIntervalForAveraging):
    """
    This function downsamples the "compliance" series so that each "timeIntervalForAveraging" has exactly a single associated value. 
    For example, if compliance was obtained with 10 Hz sampling rate, and timeIntervalForAveraging is 1 seconds, so the function averages each 10 points from the original compliance series to estimate a single value.
    Compliance series is associated to "timeSeries" in a 1-to-1 correspondence, which is used for the downsampling.
    This is to be used with a specific INEGI data-acquisition system of the LEST/UMinho.

    Parameters
    -------       
    compliance: list
        List containing the all the compliance values. It can have multiple channels as a list of lists.
    timeSeries: list
        List containing the all the time stamps of the compliance values
    timeIntervalForAveraging: float
        Time interval that will be associated to each compliance element after downsampling. Time unit must be equal to timeSeries

    Returns
    -------    
    downsampledCompliance: list
        List containing all the downsampled compliance values
    downsampledTimeSeries: list
        List containing all the downsampled time values
    """
    downsampledCompliance=[[0] for lvdtChannel in compliance]
    downsampledTimeSeries=[0]

    for indexChannel, lvdtChannel in enumerate(compliance):
        averagingCompliance=0
        averagingTime=0
        averagingCounter=0
        currentReferenceIndex=0
        for indexTime, timeValue in enumerate(timeSeries[1:-1]):
            if timeSeries[indexTime]-timeSeries[currentReferenceIndex]<=timeIntervalForAveraging:
                #This means that the points are within the timeIntervalForAveraging still
                #So we must average them
                averagingCompliance=averagingCompliance+compliance[indexChannel][indexTime]
                if indexChannel == 0:
                    #Only average time if it is the first LVDT channel
                    averagingTime=averagingTime+timeValue 
                averagingCounter=averagingCounter+1
            else:
                #So it is higher than timeIntervalForAveraging and we should stop averaging and start the process for the next point
                downsampledCompliance[indexChannel].append(averagingCompliance/averagingCounter)
                if indexChannel == 0:
                    #Only average time if it is the first LVDT channel
                    downsampledTimeSeries.append(averagingTime/averagingCounter)
                    averagingTime=timeSeries[indexTime]
                averagingCompliance=compliance[indexChannel][indexTime]
                averagingCounter=1
                currentReferenceIndex=indexTime
            #Check if it is the last eelement in timeSeries
            if indexTime == len(timeSeries):
                #It is the last element, so we should finish averaging so to avoid data loss
                downsampledCompliance[indexChannel].append(averagingCompliance/averagingCounter)
                if indexChannel == 0:
                    #Only average time if it is the first LVDT channel
                    downsampledTimeSeries.append(averagingTime/averagingCounter)

    return downsampledCompliance, downsampledTimeSeries