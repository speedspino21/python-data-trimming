import pandas as pd
import numpy as np
from tqdm import tqdm
# from fancyimpute import KNN
import KNN
import os
import savReaderWriter as srw


## << LOAD SECTION >>
## In this section, we load the data, create an emotion
## dictionary that will facilitate the indexing of
## emotion column subsets, confirm/determine the real
## ad lengths, and prepare the data so we can split it
## into smaller datasets based on ad lengths.
## FUNCTION to load .sav file.
## Input: filename is a string of the file path.
## Output: Pandas dataframe.
def loadSav(filename):
   
    with srw.SavReader(filename, returnHeader = True) as reader:
      
        header = reader.next()
        df = pd.DataFrame(reader.all())
        ## set header row as column names
        df = df.rename(columns = df.iloc[0]).drop(df.index[0])

    ## remove 'b character in front of column names due to
    ## some weird utf-8 encoding issue
    temp = [i.decode("utf-8") for i in df.columns.get_values()]
    df.columns = temp

    return(df)

## FUNCTION to create emotion dictionary, which will be
## used as way to easily index subsets of emotion columns.
## I created this because we kept changing/removing
## emotions, and this way, there is an option of deciding
## which emotions to keep as opposed to hard coding it. Also,
## who knows, maybe they'll add some other emotion in future.
## Input: data is Pandas dataframe.
## Input: emotes is list of emotions to keep.
## Output: dictionary of emotion and its associated columns.
def emoteDict(data, emotes):
    emo_dict = {}
    for emo in emotes:
        emo_name = emo.lower() + "_cols"
        emo_dict[emo_name] = [col for col in data.columns if emo in col]
    return emo_dict

## Choose which emotions to keep and create emotion dictionary.
#df = loadSav("COMBINED Respondent Level Clean Data (Sep11 2018).sav")
df = loadSav("COMBINED Respondent Level Clean Data (Sep11 2018).sav")
emotions = ["Confused", "Disgusted", "Sad", "Scared", "Surprised",
          "Happy", "Negative", "Engagement",
          "BinaryNeutral", "BinaryAttention_v2"]
emo_dict = emoteDict(df, emotions)

## Next, only keep columns of interest, such as chosen
## emotions, SourceMediaID, and Length_RE_plus1. Also, drop
## Length_RE.
keeps = emotions.copy()
keeps.extend(["SourceMediaID", "Length_RE_plus1", "TouchPointNumber"])
if "Length_RE" in df.columns:
    df = df.drop(labels = "Length_RE", axis = 1)
df = df[[col for col in df.columns for c in keeps if c in col]]

## FUNCTION to find actual lengths of ads. The Length_RE and
## Length_RE_plus1 columns are both off for some ads. Also,
## because our emotion columns are 0 indexed ("Happy_0" is the 1st 
## happy second), Length_RE is always 1 second longer than the
## actual ad length (e.g. Length_RE = 15 means there is data
## in "Happy_15", which means that this ad is actually a 16
## second ad due to the 0-indexing). NOTE: In determining
## actual ad length, we use only one of the emotion column
## subsets (e.g. Happy_0 - Happy_60), with the assumption
## that this should be enough to determine real ad length (as
## oppposed to going through all the emotions, which seems
## unnecessary and time consuming).
## Input: data is Pandas dataframe.
## Output: a Pandas dataframe, with a new column named
## "Length_Verified".
def findLengthForReal(data):
    ## get all unique ad id #s
    med_id = data["SourceMediaID"].unique()

    ## choose the 1st emotion as our test case
    test_emotion = emotions[0]

    ## loop through each ad id
    for x in med_id:
        temp = data.loc[df["SourceMediaID"] == x]
        if 0:
            ## 62 is the highest number attached to any
            ## emotion column (e.g. "Happy_62"). Change
            ## this later to do some kind of substring
            ## manipulation so that this isn't so
            ## hard coded in future.
            first_valid = 62
        else:
            first_valid = int(temp["Length_RE_plus1"]
                              .loc[temp["Length_RE_plus1"]
                              .first_valid_index()])
        test_col = test_emotion + "_" + str(first_valid)
        while temp[test_col].isnull().all() and first_valid > 0:
            first_valid = int(first_valid - 1)
            test_col = test_emotion + "_" + str(first_valid)
        ## adding + 1 because of 0-indexing
        data.loc[temp.index, "Length_Verified"] = first_valid + 1

    return(data)

df = findLengthForReal(df)
df = df.drop(labels = "Length_RE_plus1", axis = 1)
df[["Length_Verified", "SourceMediaID"]] = \
    df[["Length_Verified", "SourceMediaID"]].astype(int)

print("\nLoading & data prep for splitting/trimming completed.", u'\u2713')

## << SPLIT & TRIM SECTION >>
## In this section, we split the data into smaller datasets
## based on ad lengths. We also grab all ads that are 1-2 seconds
## more than the base length (e.g. for the 15 second dataset, we grab
## ads that are 16 and 17 seconds long), and we trim these
## down to the 15 second base length. For now, trimming
## is done in this fashion:
## 1) If subset of rows are of the base length, we're done.
## 2) If subset of rows are 1 sec more than base length, trim
##    last second.
## 3) If subset of rows are 2 secs more than base length, trim
##    last second AND first second.

## FUNCTION to split and trim the data.
## Input: data is a Pandas dataframe.
## Input: lengths is a desired list of base lengths.
## Output: dictionary of split dataframes. Key will be base
## length, while the value will be split dataframe corresponding
## to that base length.
def spliTrim(data, lengths):
    splitrim_dict = {}

    for i in lengths:
        df_temp = data.loc[data["Length_Verified"].isin(np.arange(i,i + 3))]

        ## check to see if there are any rows with selected
        ## base length or +1 or +2 seconds
        if df_temp.shape[0] == 0:
            print("There are no rows with your selected base \
                  \nlength of:", i, "secs or rows with 1 or 2 secs\
                  \nmore than your base length.")
            continue

        ## set aside columns to keep
        keep_misc = ["SourceMediaID", "Length_Verified"]
        keep_emotes = [x[0:i] for x in emo_dict.values()]
        keep_emotes_flat = [item for sublist in keep_emotes for item in sublist]
        keep_cols = keep_misc + keep_emotes_flat

        ## base length case:
        df_base = df_temp.loc[df_temp["Length_Verified"] == i]       
        df_base = df_base[keep_cols]

        ## base +1 sec case:
        df_plus1 = df_temp.loc[df_temp["Length_Verified"] == i + 1]
        df_plus1 = df_plus1[keep_cols]

        ## base +2 sec case:
        keep_emotes2 = [x[0:(i + 2)] for x in emo_dict.values()]
        keep_emotes_flat2 = [item for sublist in keep_emotes2 \
            for item in sublist]
        keep_cols2 = keep_misc + keep_emotes_flat2
        df_plus2 = df_temp.loc[df_temp["Length_Verified"] == i + 2]
        df_plus2 = df_plus2[keep_cols2]
        emote_last = [x[i + 1] for x in emo_dict.values()]
        emote_first = [x[0] for x in emo_dict.values()]
        df_plus2 = df_plus2.drop(labels = emote_last, axis = 1)
        df_plus2 = df_plus2.drop(labels = emote_first, axis = 1)
        df_plus2.columns = df_base.columns

        ## concat all 3 dfs
        df_trim = pd.concat([df_base, df_plus1, df_plus2])
        df_trim = df_trim.reset_index(drop=True)

        ## create key specific to df length
        length_key = "df" + "_" + str(i)
        splitrim_dict[length_key] = df_trim

    return splitrim_dict




# before feeding your dataframe into this class
# there are some prerequisite
# 1) make all the null values to be represented as "NA"
# # you can put np.nan in the pattern. doesn't matter
# 2) passing in data should be in aggregate form where each respondent has 2D array of emotion by time
# # this chould be found in knnHoldoutTest.py -> makeAggregate()
# before feeding your dataframe into this class
# there are some prerequisite
# 1) make all the null values to be represented as "NA"
# # you can put np.nan in the pattern. doesn't matter
# 2) passing in data should be in aggregate form where each respondent has 2D array of emotion by time
# # this chould be found in knnHoldoutTest.py -> makeAggregate()
class PatternMatcher(object): 
    def __init__(self, data, pattern, possible_values = [0.0, 0.5, 1.0], foundIndex = None,
                updatedData = None):
        self.data = data.copy()
        self.possible_values = possible_values
        self.foundIndex = foundIndex
        self.updatedData = data.copy()
        
        self.pattern = pattern
        for n,i in enumerate(self.pattern):
            if i not in possible_values: 
                self.pattern[n] = "NA"
        

    # returns found index in the form of list
    # the list contains list, where the nested list is the coordination of the pattern found
    # ex: [3, 0, 0] --> pattern happend in 3 indexed respondent, at 0 indexed emotion, at 0 index
    def returnIndex(self):
        pattLength = len(self.pattern)
        indexMaps = []
        for index, row in self.updatedData.iterrows():
            values = row['Value']           
            for valn, val in enumerate(values):
                for n,i in enumerate(val):
                    matches = None
                    endLength = len(val) - pattLength + 1  
                    if ((i == self.pattern[0]) & (n < endLength)):
                        j = 1
                        isMatch = True
                        while ((j < pattLength) & isMatch):
                            if (val[j+n] == self.pattern[j]):
                                j += 1
                                #print([index, valn, n, val[j], self.pattern[j], j])
                            else:
                                isMatch = False
                        if (isMatch): 
                            matches = [index, valn, n]
                        if (matches):
                            indexMaps.append(matches)
        self.foundIndex = indexMaps
        return indexMaps
    
    def imputePattern(self, desiredImputation):
        if (len(desiredImputation) != len(self.pattern)):
            print("Desired imputation and pattern does not have the same length")
            print("They need to be in same length. ")
            print("Desired Imputation Length:", len(desiredImputation), "Pattern:", len(self.pattern))
            return 0 
                
        for i in tqdm(self.foundIndex): 
            j = 0 
            while (j < len(self.pattern)): 
                self.updatedData.iloc[i[0]].Value[i[1]][i[2] + j] = desiredImputation[j]
                j += 1
        return self.updatedData
        

 ## Usage
 # df = df.fillna("NA")
 # patt = patternMatcher(makeAggregate(df), pattern = [1, np.nan, np.nan, 1])
 # indexFound = patt.returnIndex()
 # ## patt.pattern, len(patt.foundIndex)
 # imputed = patt.imputePattern([1,1,1,1])
 # ##imputed       


### We need to add emotions, if we decide to use different emotions other than the ones in 
### `emotion_list`
def makeAggregate(df): 
# 	"""
# 	makeAggregate returns a dataframe with just one column 
# 	-- it aggregates all n number of second per emotion m 
# 	and makes m x n matrix for each column, that goes to "Value" column
# 	this is easier to line up all the emotion value per second
# 	"""
    emotion_list = ["Confused", "Disgusted", "Sad", "Scared", "Surprised", "Happy", "Negative", "Engagement",\
            "BinaryNeutral", 'BinaryAttention']
    
    eevg = {}
    for i in emotion_list:
        aa = []
        for j in df.columns:
            if (i in j):
                aa.append(j)
        eevg[i] = aa

    confused = pd.DataFrame(df[eevg['Confused']].apply(lambda x: x.tolist(), axis=1), columns=["Confused"])
    confused.index = df.index

    disgusted = pd.DataFrame(df[eevg["Disgusted"]].apply(lambda x: x.tolist(), axis = 1), columns = ["Disgusted"])
    disgusted.index = df.index

    sad = pd.DataFrame(df[eevg["Sad"]].apply(lambda x: x.tolist(), axis=1), columns = ["Sad"])
    sad.index = df.index

    scared = pd.DataFrame(df[eevg["Scared"]].apply(lambda x: x.tolist(), axis=1), columns = ["Scared"])
    scared.index = df.index

    surprised = pd.DataFrame(df[eevg["Surprised"]].apply(lambda x: x.tolist(), axis=1), columns = ["Surprised"])
    surprised.index = df.index

    happy = pd.DataFrame(df[eevg["Happy"]].apply(lambda x: x.tolist(), axis=1), columns = ["Happy"])
    happy.index = df.index

    negative = pd.DataFrame(df[eevg["Negative"]].apply(lambda x: x.tolist(), axis=1), columns = ["Negative"])
    negative.index = df.index

    engagement = pd.DataFrame(df[eevg["Engagement"]].apply(lambda x: x.tolist(), axis=1), columns = ["Engagement"])
    engagement.index = df.index
    
    neutral = pd.DataFrame(df[eevg["BinaryNeutral"]].apply(lambda x: x.tolist(), axis=1), columns = ["BinaryNeutral"])
    neutral.index = df.index

    BinaryAttention = pd.DataFrame(df[eevg["BinaryAttention"]].apply(lambda x: x.tolist(), axis=1), columns = ["BinaryAttention"])
    BinaryAttention.index = df.index
    
    total_df = [confused, disgusted, sad, scared, surprised, happy, negative, engagement, neutral, BinaryAttention]

    df_final = pd.concat(total_df, axis =1 )
    aggregated = pd.DataFrame(df_final.apply(lambda x: x.tolist(), axis=1))

    aggregated.columns = ["Value"]

    return aggregated


## imputationPipeline that actually does imputation with the functions and classes above

## decide which base lengths we want
lengths = [15, 20, 30, 60]

## split and trim the data
dfs_trimmed = spliTrim(df, lengths)

print("Splitting and trimming is completed.", u'\u2713')

## create new directory for trimmed data
if not os.path.isdir("./data-trimmed"):
    os.mkdir("data-trimmed")

## save trimmed dataframes into new directory
print("Writing dfs to csv files:")
for key, df in dfs_trimmed.items():
    fname = key + "_trimmed.csv"
    fpath = "./data-trimmed/" + fname
    df.to_csv(path_or_buf = fpath, index = False)
    ## the 1st empty space in the print statement below is
    ## intentional for text to line up
    print(" Trimmed dataframe:", fname, "written to the\n",
          "data-trimmed directory.", u'\u2713', "\n")

print("All loading, splitting, and trimmimg completed.", u'\u2713')


class imputationPipeline(object):
    def __init__(self, path, pipeline, lengths, run_knn=True):
        self.path = path
        self.data = loadSav(path)
        ## drop all rows with all NaNs
        self.df = self.df.dropna(axis = 0, how = 'all')

		## organize data by ad id and reported length
        self.df = self.df.sort_values(["SourceMediaID","Length_RE_plus1"]).reset_index(drop = True)
        self.pipeline = pipeline
        self.run_knn = run_knn
        # data is in the form of dataframe
        # pipeline is in the form of dictionary

        ## Choose which emotions to keep and create emotion dictionary.
        emotions = ["Confused", "Disgusted", "Sad", "Scared", "Surprised","Happy", "Negative", "Engagement","BinaryNeutral", "BinaryAttention_v2"]
        emo_dict = emoteDict(self.df, emotions)

		## Next, only keep columns of interest, such as chosen
		## emotions, SourceMediaID, and Length_RE_plus1. Also, drop
		## Length_RE.
        keeps = emotions.copy()
        keeps.extend(["SourceMediaID", "Length_RE_plus1", "TouchPointNumber"])
        if "Length_RE" in self.df.columns:
            self.df = self.df.drop(labels = "Length_RE", axis = 1)
        self.df = self.df[[col for col in self.df.columns for c in keeps if c in col]]

        self.df = findLengthForReal(self.df)
        self.df = self.df.drop(labels = "Length_RE_plus1", axis = 1)
        self.df[["Length_Verified", "SourceMediaID"]] = \
            self.df[["Length_Verified", "SourceMediaID"]].astype(int)

        self.df = spliTrim(self.df, lengths)
    
    def runPipeline(self, seeDiff=True, k =10):
        filtered = self.data.filter(regex=('_\d')).copy()
        filtered.fillna('NA', inplace=True)
        
        agg_filtered = makeAggregate(filtered)
        
        print("Running pipeline length", len(pipeline))
        #print("Start distribution", getDistribution(agg_filtered))
        
        distributions= []
        for number, imputation in enumerate(pipeline):
            print("Imputation number", number+1)
            print("Pattern: ", imputation[0], "Imputation: ", imputation[1])
            patternMatcher = PatternMatcher(agg_filtered, imputation[0])
            indexes = patternMatcher.returnIndex()
            print(len(indexes), "of pattern found")
            agg_filtered = patternMatcher.imputePattern(imputation[1])

            #print("Distribution of Imputation number", number+1)
            #print('\n',getDistribution(agg_filtered))
            distributions.append(getDistribution(agg_filtered))
        
        
        retVal = unAggregate(agg_filtered, self.data)
        retVal = retVal.replace({'NA':np.nan})

        if (self.run_knn):
            for_knn = retVal
            for_knn = for_knn.filter(regex=('_\d'))
            knn_imputed = KNN(k=k).complete(for_knn)
            knn_imputed = pd.DataFrame(knn_imputed)
            knn_imputed.columns = for_knn.columns
            knn_imputed = knn_imputed.applymap(bar)
            
            diff = set(self.data.columns).difference(set(self.data.filter(regex=('_\d')).columns))
            for i in diff: 
                knn_imputed[i] = self.data[i]
            retVal = knn_imputed
            
        if (seeDiff): 
            return retVal, distributions
        else:
            return retVal

# This will drop respondents with more than 20% of NaN values
def dropRespondents(df_main,perc=0.8):
    
    print(df_main.shape)
    df_target = df_main.filter(regex=('_\d')).copy()
    df_target['nan_perc'] = df_target.isnull().sum(axis=1)/ df_target.shape[1]
    print("--------123123----")
    print(df_main)
    print("------------")
    df_target["Country"] = df_main['Country']
    df_target['Length_Verified'] = df_main['Length_Verified']
    df_target['SourceMediaID'] = df_main['SourceMediaID']
    drop_index = list(df_target[df_target['nan_perc']>=perc].index)
    df_target = df_target.drop(drop_index)
    df_target.reset_index(inplace=True)
    df_target.rename(columns={'index':'original_index'}, inplace=True)
    print("original shape:", df_main.shape, "new shape:", df_target.shape, "after dropping:", len(drop_index))
    return df_target


def unAggregate(aggregate_df, original_df): 
    rows = []
    for index, row in aggregate_df.iterrows(): 
        rows.append(np.array(row['Value']).flatten())
    retVal = pd.DataFrame(rows)
    
    diff = set(original_df.columns).difference(set(original_df.filter(regex=('_\d')).columns))
    retVal.columns = original_df.filter(regex=('_\d')).columns
    
    for additionalColumn in diff:
        retVal[additionalColumn] = original_df[additionalColumn]
    
    return 

def afunction(x):
    if (x == "NA"):
        return np.nan
    else:
        return x


## Need to change this emotions if the emotions in agg changes
def getDistribution(agg_DataFrame): 
    ed = {
    'Confused': {0.0:0, 0.5:0, 1.0:0},
    'Sad': {0.0:0, 0.5:0, 1.0:0},
    'Disgusted': {0.0:0, 0.5:0, 1.0:0},
    'Surprised': {0.0:0, 0.5:0, 1.0:0},
    'Negative': {0.0:0, 0.5:0, 1.0:0},
    'Happy': {0.0:0, 0.5:0, 1.0:0},
    'Scared': {0.0:0, 0.5:0, 1.0:0},
    'BinaryNeutral': {0.0:0, 0.5:0, 1.0:0},
    'Engagement': {0.0:0, 0.5:0, 1.0:0},
    'BinaryAttention': {0.0:0, 0.5:0, 1.0:0}
    }
    for i in agg_DataFrame.Value.values:

        ed['Confused'][0.0] += i[0].count(0)
        ed['Confused'][0.5] += i[0].count(0.5)
        ed['Confused'][1.0] += i[0].count(1.0)

        ed['Disgusted'][0.0] += i[1].count(0)
        ed['Disgusted'][0.5] += i[1].count(0.5)
        ed['Disgusted'][1.0] += i[1].count(1.0)

        ed['Sad'][0.0] += i[2].count(0)
        ed['Sad'][0.5] += i[2].count(0.5)
        ed['Sad'][1.0] += i[2].count(1.0)

        ed['Scared'][0.0] += i[3].count(0)
        ed['Scared'][0.5] += i[3].count(0.5)
        ed['Scared'][1.0] += i[3].count(1.0)

        ed['Surprised'][0.0] += i[4].count(0)
        ed['Surprised'][0.5] += i[4].count(0.5)
        ed['Surprised'][1.0] += i[4].count(1.0)

        ed['Happy'][0.0] += i[5].count(0)
        ed['Happy'][0.5] += i[5].count(0.5)
        ed['Happy'][1.0] += i[5].count(1.0)

        ed['Negative'][0.0] += i[6].count(0)
        ed['Negative'][0.5] += i[6].count(0.5)
        ed['Negative'][1.0] += i[6].count(1.0)

        ed['Engagement'][0.0] += i[7].count(0)
        ed['Engagement'][0.5] += i[7].count(0.5)
        ed['Engagement'][1.0] += i[7].count(1.0)

        ed['BinaryNeutral'][0.0] += i[8].count(0)
        ed['BinaryNeutral'][0.5] += i[8].count(0.5)
        ed['BinaryNeutral'][1.0] += i[8].count(1.0)
        
        ed['BinaryAttention'][0.0] += i[9].count(0)
        ed['BinaryAttention'][0.5] += i[9].count(0.5)
        ed['BinaryAttention'][1.0] += i[9].count(1.0)


    return ed



########## Example of generating ALL different types of data 
# do = [ '15', '20', '30', '60']
# for sec in do: 
#     print('***Running', sec,'Sec***\n\n')


#     df_main = pd.read_csv('/Users/sherly.kim/Desktop/BA_v2/df_'+sec+'_trimmed.csv') # get the original data
#     df_main = dropRespondents(df_main)
#     pipeline = [([0, np.nan, 0],[0,0,0]),
#             ([1, np.nan, 0.5], [1,1,0.5]),
#              ([1, np.nan, 1], [1,1,1]),
#             ([0, np.nan, np.nan, 0], [0,0,0,0]),
#              ([1, np.nan, np.nan, 1], [1,1,1,1])]
		##These pipeline is suggested from ISC 

#     just_incase = df_main.copy()

#     nodrop_pipeline = imputationPipeline(df_main, pipeline, run_knn = True)
#     nodrop_df, _ = nodrop_pipeline.runPipeline()
