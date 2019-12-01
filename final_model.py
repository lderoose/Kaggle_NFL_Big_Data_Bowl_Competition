import pandas as pd
import numpy as np
import datetime
import time
import math
import datetime
import lightgbm
import warnings
from datetime import date
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold

from kaggle.competitions import nflrush
env = nflrush.make_env()

#Load data
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



################################################################################################################################
######################################################### FUNCTIONS ############################################################
################################################################################################################################

def preprocessing(df):
    
    def cleanAbbr(df):
        df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
        df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"
        df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
        df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"
        df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
        df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"
        df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
        df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"
        return 

    def cleanPosition(df):
        df.loc[df.Position == "T", 'Position'] = "OT"
        df.loc[df.Position == "ILB", 'Position'] = "LB"
        df.loc[df.Position == "SAF", 'Position'] = "S"
        return 
    
    def groupPosition(df):
        df.loc[df.Position == "RB", 'Position'] = "B"
        df.loc[df.Position == "FB", 'Position'] = "B"
        df.loc[df.Position == "HB", 'Position'] = "B"
        df.loc[df.Position == "OT", 'Position'] = "OL"
        df.loc[df.Position == "G", 'Position'] = "OL"
        df.loc[df.Position == "OG", 'Position'] = "OL"
        df.loc[df.Position == "OLB", 'Position'] = "LB"
        df.loc[df.Position == "MLB", 'Position'] = "LB"
        df.loc[df.Position == "NT", 'Position'] = "DL"
        df.loc[df.Position == "DE", 'Position'] = "DL"
        df.loc[df.Position == "DT", 'Position'] = "DL"
        df.loc[df.Position == "CB", 'Position'] = "DB"
        df.loc[df.Position == "S", 'Position'] = "DB"
        df.loc[df.Position == "FS", 'Position'] = "DB"
        df.loc[df.Position == "SS", 'Position'] = "DB"
        return 
    
    def updateOrientation2017(df):
        df.loc[df['Season'] == 2017, 'Orientation'] = np.mod(90 + df.loc[df['Season']==2017, 'Orientation'], 360)
        return
    
    def standardizeGame(df):
        #paste on https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
        df['ToLeft'] = df.PlayDirection == "left"
        df['IsBallCarrier'] = df.NflId == df.NflIdRusher 
        df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
        df['TeamOnOffense'] = "home"
        df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
        df['IsOnOffense'] = df.Team == df.TeamOnOffense 
        df['YardLine_std'] = 100 - df.YardLine 
        df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
                  'YardLine_std'
                 ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
                  'YardLine']
        df['X_std'] = df.X
        df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] 
        df['Y_std'] = df.Y
        df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y'] 
        df['Orientation_std'] = -90 + df.Orientation
        df.loc[df.ToLeft, 'Orientation_std'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation_std'], 360)
        df['Dir_std'] = df.Dir_rad
        df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
        df.drop(
            [
            'X', 'Y', 'Orientation', 'Dir', 'PlayerCollegeName', 'DisplayName', 'NflIdRusher', 'ToLeft',
            'Team', 'IsBallCarrier', 'NflId', 'WindSpeed', 'WindDirection', 'PossessionTeam',
            'FieldPosition', 'PlayDirection', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location',
            'WindDirection', 'TeamOnOffense', 'JerseyNumber', 'IsOnOffense',
            ], axis=1, inplace=True)
        return 

    cleanAbbr(df)
    cleanPosition(df)
    groupPosition(df)
    updateOrientation2017(df)
    standardizeGame(df)
    return


#----------------------------------------
def convertObjectFeatures(df):

    def convertStadiumType(stadiumType):
        wordForOpenRoof = ['Outdoor', 'Outdoors', 'Open', 'Domed, open', 'Oudoor', 'Retr. Roof-Open', 'Outddors',
                          'Heinz Field', 'Outdoor Retr Roof-Open', 'Retr. Roof - Open', 'Ourdoor', 'Outside',
                          'Indoor, Open Roof', 'Outdor', 'Cloudy', 'Domed, Open']   
        wordForClosedRoof = ['Indoor', 'Indoors', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Domed, closed',
                            'Closed Dome', 'Dome, closed', 'Retr. Roof Closed', 'Indoor, Roof Closed',
                            'Dome', 'Domed']   
        unknownRoof = ['Retractable Roof', 'Bowl', 'N/A Indoors']
        if stadiumType in wordForOpenRoof:
            return 1
        elif stadiumType in wordForClosedRoof:
            return 0
        elif stadiumType in unknownRoof:
            return np.nan
        else:
            print('error stadiumType : ', stadiumType)
            return np.nan

    def convertGameWeather(weather):
        rain = [
            'Rainy', 'Rain Chance 40%', 'Showers',
            'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
            'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain'
        ]
        overcast = [
            'Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',
            'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
            'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
            'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
            'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
            'Partly Cloudy', 'Cloudy', 'overcast', 'Partly cloudy and mild', 'Breezy', 'Light rain',
            'partly cloudy'
        ]
        clear = [
            'Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
            'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
            'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
            'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
            'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
            'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', 'Sunny, Windy', 'clear',
            'Mostly Clear'
        ]
        snow  = ['Heavy lake effect snow', 'Snow']
        none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']
        if weather in rain:
            return 3
        elif weather in overcast:
            return 1
        elif weather in clear:
            return 0
        elif weather in snow:
            return 5
        elif weather in none:
            return np.nan
        elif pd.isna(weather):
            return np.nan    
        else:
            print('error weather : ', weather)
            return weather

    def convertTurf(turf):
        wordForNatural = ['Natural', 'Natural Grass','Natural grass', 'Naturall Grass', 'natural grass']
        wordForArtifical = ['Artificial', 'UBU Speed Series-S5-M', 'A-Turf Titan', 'UBU Sports Speed S5-M',
                           'FieldTurf360', 'DD GrassMaster', 'Twenty-Four/Seven Turf', 'SISGrass', 'FieldTurf 360',
                           'Artifical', 'Field Turf', 'FieldTurf', 'Field turf']
        unknownTurf = ['Grass', 'grass']
        if turf in wordForNatural:
            return 'natural_turf'
        elif turf in wordForArtifical:
            return 'artificial_turf'
        elif turf in unknownTurf:
            return 'unknownTurf'
        else:
            print('error turf : ', turf)
            return np.nan
    
    #TO DO : fix label encoder
    from sklearn import preprocessing
    LE = preprocessing.LabelEncoder()    
    df['PlayerWeight'] = round(df['PlayerWeight'].apply(lambda x: x*.453592), 2)
    df['PlayerHeight'] = round(df['PlayerHeight'].apply(lambda x: int(x[0])*30.48 + int(x[2:])*2.54), 2)
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    df['PlayerBirthDate'] = round(df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/(60*60*24*365.25), axis=1), 2)
    df['OffenseFormation'].fillna(df['OffenseFormation'].value_counts().index[0], inplace=True) #max card obs
    df['OffenseFormation'] = LE.fit_transform(df['OffenseFormation'].values.tolist())
    df['StadiumType'] = df['StadiumType'].apply(convertStadiumType)
    df['GameWeather'] = df['GameWeather'].apply(convertGameWeather)
    df['Turf'] = df['Turf'].apply(convertTurf)
    df['Turf'] = LE.fit_transform(df['Turf'].values.tolist())
    df['OffensePersonnel'].fillna(df['OffensePersonnel'].value_counts().index[0], inplace=True)
    df['OffensePersonnel'] = LE.fit_transform(df['OffensePersonnel'].values.tolist())
    df['DefensePersonnel'].fillna(df['DefensePersonnel'].value_counts().index[0], inplace=True)
    df['DefensePersonnel'] = LE.fit_transform(df['DefensePersonnel'].values.tolist())
    return 


#----------------------------------------
def createColumnsName(role, features):
    """ add role for each element in features """
    return [col + '_' + role for col in features]
    
#----------------------------------------
def createMultipleColumnsName(role, features, nbPlayers):
    """ add role + number for each element in features and repeat this operation for the nbPlayers """
    def createFeatures(role, features, nbPlayers=1):
        return [elem+'_'+role+str(nbPlayers) for elem in features]
    l = []
    for pl in range(1, nbPlayers+1):
        l = l + createFeatures(role, features, str(pl))
    return l


#----------------------------------------
def createPlayersPositionTrain(df, features):
    
    def reshapingColumnStructure(df, features):
        return pd.DataFrame(df.values.reshape(-1, int(df.shape[1]/22)), columns = features)
    
    def dropPositionCols(df):
        """ deletes the column 'Position' that has become useless"""
        list_cols = []
        for col in df.columns.to_list():
            if 'Position' in col:
                list_cols.append(col)
        df.drop(list_cols, axis = 1, inplace=True)
        return df
    
    def createUniquePlayer(df, position, features):
        """ used to create features for players which are suppose to be unique : Quarterback and Center """
        pos = df['Position'].apply(lambda x: 1 if x == position else None).dropna()
        pos = pos.index.to_list() #first element if many
        if pos:
            return df.loc[pos[0], features].values
        else:
            return np.empty((len(features)), dtype='object')
        
    def createMultiplePlayers(df, position, features, variable, maxCols, reference):
        """ used to create features for players which are suppose to not be unique : not Quarterback or Center """

        def searchMultiple(df, position, features):
            pos = df['Position'].apply(lambda x: 1 if x == position else None).dropna()
            pos = pos.index.to_list()
            if pos:
                return df.loc[pos, features]
            else:
                return pd.DataFrame([np.empty((len(features)), dtype='object')], columns = features)

        def supPlayer(df, dfReference, variable):
            """ extract players who are placed upper the Quarterbarck in regard to 'variable' """
            if df.empty == False:
                df_sup = df[df[variable] > dfReference.values[0]]
                df_sup.sort_values(by = variable, axis=0, ascending = False, inplace = True)
                df_sup.index = range(0, 2*df_sup.shape[0], 2) #even
                return df_sup
            else:
                return df

        def infPlayer(df, dfReference, variable):
            """ extract players who are placed under the Quarterbarck in regard to 'variable' """
            if df.empty == False:
                df_inf = df[df[variable] <= dfReference.values[0]]
                df_inf.sort_values(by = variable, axis = 0, ascending = True, inplace = True)
                df_inf.index = range(1, 2*df_inf.shape[0], 2) #uneven
                return df_inf
            else: 
                return df

        def fusion(df_sup, df_inf):
            """ join the dataframes in the good order : 1st = higher, 2nd = lower, 3rd = second higher ... """
            if df_sup.empty:
                if df_inf.empty == False:
                    return df_inf
                else:
                    return df_sup
            else:
                if df_inf.empty == False:
                    return pd.concat([df_sup, df_inf], axis = 0, ignore_index = False).sort_index()
                else:
                    return df_sup

        def fillArray(dfFusion, features):
            """ fill a part of max empty dataframe """
            dfEmpty = pd.DataFrame( maxCols*[np.empty((len(features)), dtype = 'object')], columns = features )
            if dfFusion.empty == False:
                dfEmpty.loc[dfFusion.index, features] = dfFusion
            return dfEmpty

        def reshapeArray(dfFill, features, maxCols):
            """ reshaping data into a big row and drop columns which are empty (for test -> predict) """
            array = pd.DataFrame(dfFill.values.reshape(-1, maxCols*len(features)),
                                 columns = createMultipleColumnsName(role = position,
                                                                     features = features, nbPlayers = maxCols))
            return array
        
        dfSup = supPlayer(df = searchMultiple(df = df, position = position, features = features),
                          dfReference = reference, variable = variable)
        dfInf = infPlayer(df = searchMultiple(df = df, position = position, features = features),
                          dfReference = reference, variable = variable)
        df = fusion(df_sup = dfSup, df_inf = dfInf)
        df = fillArray(dfFusion = df, features = features)
        df = reshapeArray(dfFill = df , features = features, maxCols = maxCols)
        
        return df
    
    df = pd.DataFrame(df).transpose()
    df = reshapingColumnStructure(df, features)
    df_qb = pd.DataFrame([createUniquePlayer(df = df, position = 'QB', features = features)],
                         columns = createColumnsName(role = 'QB', features = features))
    df_c = pd.DataFrame([createUniquePlayer(df = df, position = 'C', features = features)], 
                        columns = createColumnsName(role = 'C', features = features))
    df_wr = createMultiplePlayers(df = df, position = 'WR', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_te = createMultiplePlayers(df = df, position = 'TE', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_b = createMultiplePlayers(df = df, position = 'B', features = features, variable = 'X_std',
                                  maxCols = 22, reference = df_qb['X_std_QB'])
    df_ol = createMultiplePlayers(df = df, position = 'OL', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_dl = createMultiplePlayers(df = df, position = 'DL', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_lb = createMultiplePlayers(df = df, position = 'LB', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_db = createMultiplePlayers(df = df, position = 'DB', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    
    df_full = pd.concat([df_qb, df_c, df_wr, df_te, df_b, df_ol, df_dl, df_lb, df_db], axis = 1)
#     df_full = dropPositionCols(df_full)
    return df_full

#----------------------------------------
def keepNonPlayersFeaturesTrain(df, features):
    """ keep the features that are not relative to players """
    df_base = df.drop(features, axis = 1, inplace = False).drop_duplicates('PlayId', inplace = False).reset_index(drop = True)
    return df_base

#----------------------------------------
def reshapingRowStructure(df, features):
    return pd.DataFrame(df[features].values.reshape(-1, 22*len(features)), columns=[features*22])

#----------------------------------------
def featuresEngineering(df):
    
    def timeToEndQuarter(clock):
        try:
            t=time.strptime(clock, '%H:%M:%S')
            return t.tm_hour*60 + t.tm_min
        except:
            return np.nan

    def timeToEndMatch(quarter, clock):
        try:
            t=time.strptime(clock, '%H:%M:%S')
            if quarter>1:
                return (3600 - ((quarter-1) *900 + 900-(t.tm_hour*60 + t.tm_min)))
            else:
                return (3600 - 900+(t.tm_hour*60 + t.tm_min))
        except:
            return np.nan
        
    def dropUseless(df, col_to_remove):
        df.drop(col_to_remove, axis= 1, inplace = True)
        return
    
    def convertToCategorical(df, categorical_features):
        """ convert type of categorical columns """
        for col in categorical_features:
            df[col] = df[col].astype('category')
            df[col] = pd.Categorical(df[col].cat.codes+1)
            df[col] = df[col].astype(np.int64)
        return 
        
    df['TimeToEndQuarter'] = df['GameClock'].apply(timeToEndQuarter)
    df['TimeStartQuarterToNow'] = 900-df['TimeToEndQuarter']
    df['TimeToEndMatch'] = df[['Quarter', 'GameClock']].apply(lambda x: timeToEndMatch(x['Quarter'], x['GameClock']), axis=1)
    df['TimeStartMatchToNow'] = 3600-df['TimeToEndMatch']
    df['DefendersInTheBox'].fillna(df['DefendersInTheBox'].value_counts().index[0], inplace=True)
    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
    
    dropUseless(df, ['GameClock', 'TimeHandoff', 'TimeSnap']) #peut-être GameId et PlayId
    convertToCategorical(df, ['Season', 'Quarter', 'Down', 'OffenseFormation', 'OffensePersonnel', 
                              'DefendersInTheBox', 'DefensePersonnel', 'Week', 'StadiumType', 'Turf', 
                              'GameWeather'])
    return

def dropPositionCols(df):
    """ deletes the column 'Position' that has become useless"""
    list_cols = []
    for col in df.columns.to_list():
        if 'Position' in col:
            list_cols.append(col)
    df.drop(list_cols, axis = 1, inplace=True)
    return df

#----------------------------------------
def createPlayersPositionTest(df, features):
    
    def keepNonPlayersFeatures(df, features):
        """ keep the features that are not relative to players """
        df_base = df.drop(features, axis = 1, inplace = False).drop_duplicates('GameId', inplace = False).reset_index(drop=True)
        return df_base
    
    def dropPositionCols(df):
        """ deletes the column 'Position' that has become useless"""
        list_cols = []
        for col in df.columns.to_list():
            if 'Position' in col:
                list_cols.append(col)
        df.drop(list_cols, axis = 1, inplace=True)
        return df
    
    def createColumnsName(role, features):
        """ add role for each element in features """
        return [col + '_' + role for col in features]
    
    def createMultipleColumnsName(role, features, nbPlayers):
        """ add role + number for each element in features and repeat this operation for the nbPlayers """
        def createFeatures(role, features, nbPlayers=1):
            return [elem+'_'+role+str(nbPlayers) for elem in features]
        l = []
        for pl in range(1, nbPlayers+1):
            l = l + createFeatures(role, features, str(pl))
        return l
    
    def createUniquePlayer(df, position, features):
        """ used to create features for players which are suppose to be unique : Quarterback and Center """
        pos = df['Position'].apply(lambda x: 1 if x == position else None).dropna()
        pos = pos.index.to_list() #first element if many
        if pos:
            return df.loc[pos[0], features].values
        else:
            return np.empty((len(features)), dtype='object')
        
    def createMultiplePlayers(df, position, features, variable, maxCols, reference):
        """ used to create features for players which are suppose to not be unique : not Quarterback or Center """

        def searchMultiple(df, position, features):
            pos = df['Position'].apply(lambda x: 1 if x == position else None).dropna()
            pos = pos.index.to_list()
            if pos:
                return df.loc[pos, features]
            else:
                return pd.DataFrame([np.empty((len(features)), dtype='object')], columns = features)

        def supPlayer(df, dfReference, variable):
            """ extract players who are placed upper the Quarterbarck in regard to 'variable' """
            if df.empty == False:
                df_sup = df[df[variable] > dfReference.values[0]]
                df_sup.sort_values(by = variable, axis=0, ascending = False, inplace = True)
                df_sup.index = range(0, 2*df_sup.shape[0], 2) #even
                return df_sup
            else:
                return df

        def infPlayer(df, dfReference, variable):
            """ extract players who are placed under the Quarterbarck in regard to 'variable' """
            if df.empty == False:
                df_inf = df[df[variable] <= dfReference.values[0]]
                df_inf.sort_values(by = variable, axis = 0, ascending = True, inplace = True)
                df_inf.index = range(1, 2*df_inf.shape[0], 2) #uneven
                return df_inf
            else: 
                return df

        def fusion(df_sup, df_inf):
            """ join the dataframes in the good order : 1st = higher, 2nd = lower, 3rd = second higher ... """
            if df_sup.empty:
                if df_inf.empty == False:
                    return df_inf
                else:
                    return df_sup
            else:
                if df_inf.empty == False:
                    return pd.concat([df_sup, df_inf], axis = 0, ignore_index = False).sort_index()
                else:
                    return df_sup

        def fillArray(dfFusion, features):
            """ fill a part of max empty dataframe """
            dfEmpty = pd.DataFrame( maxCols*[np.empty((len(features)), dtype = 'object')], columns = features )
            if dfFusion.empty == False:
                dfEmpty.loc[dfFusion.index, features] = dfFusion
            return dfEmpty

        def reshapeArray(dfFill, features, maxCols):
            """ reshaping data into a big row and drop columns which are empty (for test -> predict) """
            array = pd.DataFrame(dfFill.values.reshape(-1, maxCols*len(features)),
                                 columns = createMultipleColumnsName(role = position,
                                                                     features = features, nbPlayers = maxCols))
            return array
        
        dfSup = supPlayer(df = searchMultiple(df = df, position = position, features = features),
                          dfReference = reference, variable = variable)
        dfInf = infPlayer(df = searchMultiple(df = df, position = position, features = features),
                          dfReference = reference, variable = variable)
        df = fusion(df_sup = dfSup, df_inf = dfInf)
        df = fillArray(dfFusion = df, features = features)
        df = reshapeArray(dfFill = df , features = features, maxCols = maxCols)
        
        return df
    
    df_base = keepNonPlayersFeatures(df = df, features = features)
    df_qb = pd.DataFrame([createUniquePlayer(df = df, position = 'QB', features = features)],
                         columns = createColumnsName(role = 'QB', features = features))
    df_c = pd.DataFrame([createUniquePlayer(df = df, position = 'C', features = features)], 
                        columns = createColumnsName(role = 'C', features = features))
    df_wr = createMultiplePlayers(df = df, position = 'WR', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_te = createMultiplePlayers(df = df, position = 'TE', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_b = createMultiplePlayers(df = df, position = 'B', features = features, variable = 'X_std',
                                  maxCols = 22, reference = df_qb['X_std_QB'])
    df_ol = createMultiplePlayers(df = df, position = 'OL', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_dl = createMultiplePlayers(df = df, position = 'DL', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_lb = createMultiplePlayers(df = df, position = 'LB', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    df_db = createMultiplePlayers(df = df, position = 'DB', features = features, variable = 'Y_std',
                                  maxCols = 22, reference = df_qb['Y_std_QB'])
    
    df_full = pd.concat([df_base, df_qb, df_c, df_wr, df_te, df_b, df_ol, df_dl, df_lb, df_db], axis = 1)
    df_full = dropPositionCols(df_full)
    return df_full
    
    
################################################################################################################################
##################################################### DATA PREPROCESSING #######################################################
################################################################################################################################

print('Start Data preprocessing ...')    
preprocessing(df = train)
print('... End preprocessing part 1 ...')   
convertObjectFeatures(df = train)
print('... End convert Object Features ...')   


print('... Start creation players ...')  
features_players = [ 'S', 'A', 'X_std', 'Y_std', 'Orientation_std', 'PlayerBirthDate', 'PlayerWeight', 
                    'PlayerHeight', 'Position', 'Dir_std', 'Dis']

maxCols = 22
nb_col = maxCols*7*len(features_players) + 2*len(features_players)
cols_name = createColumnsName(role = 'QB', features = features_players) + \
            createColumnsName(role = 'C', features = features_players) + \
            createMultipleColumnsName(role = 'WR', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'TE', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'B', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'OL', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'DL', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'LB', features = features_players, nbPlayers = maxCols) + \
            createMultipleColumnsName(role = 'DB', features = features_players, nbPlayers = maxCols)

base = keepNonPlayersFeaturesTrain(df = train, features = features_players)
train = reshapingRowStructure(df = train, features = features_players)

train = np.concatenate(train.apply(lambda x: createPlayersPositionTrain(x, features_players), axis = 1))
train = pd.concat([base, pd.DataFrame(train, columns = cols_name)], axis = 1)

print('... End creation players ...')  
train = dropPositionCols(train)
featuresEngineering(train)

print('... End Features engineering ...') 
print('... End Data preprocessing') 


################################################################################################################################
####################################################### MACHINE LEARNING #######################################################
################################################################################################################################

# Custom Evaluation Function :
def convertValToArrayEval(arrayOfValue):
    arrayOfArray = np.zeros(shape=(arrayOfValue.shape[0], 199))
    for i,yard in enumerate(arrayOfValue):  
        arrayOfArray[i, int(yard):] = np.ones(shape=(1, 199-int(yard)))
    return arrayOfArray

def Crps(true, pred):
    """format : (#prediction, 199)"""
    return np.sum((pred - true)**2)/(199*pred.shape[0])

def evaluationFunction(preds, true, coeff=1000):
    """ 1000 * CRPS Evaluation function """
    labels = true.get_label()
    labels = convertValToArrayEval(labels)
    p = pd.DataFrame(preds.reshape(199, labels.shape[0]))
    p = np.transpose(p.cumsum())
    return str(coeff)+"*CRPS", coeff*Crps(labels, p.values), False


# Data preparation 
train.dropna(axis = 1, how = 'all', inplace = True)
col_X = [col for col in train.columns.to_list() if col not in ["Yards", 'YardLine', 'GameId']]
train['Yards'] = pd.DataFrame(train['Yards'] + 99, columns = ['Yards'])

# Parametres LightGbm
param_classifieur = {'boosting_type': 'gbdt', 
                     'colsample_bytree': .7, 
                     'learning_rate': 0.01,
                     'max_depth': 6, 
                     'min_sample_leaf':20,
                     'subsample':1, 
                     'num_leaves': 30,
                     'objective': 'multiclass',
                     'num_class':199,
                     'random_state': None,
                     'reg_alpha': 5, # regularization
                     'reg_lambda': 5, # regularization
                     'subsample_for_bin': 1000,
                     'subsample_freq':1,
                     'max_bin':30,
                     'verbose':-1,
                     'metric':'None',
                     'boost_from_average':True, #IMPORTANT !
                     'use_missing':True,
                     'is_unbalance':True}

kf_train = KFold(n_splits = 2, shuffle = False)
kf_validation = KFold(n_splits = 2, shuffle = False)

for train_index, test1_index in kf_train.split(train[train['Season'] == 2]):
    for validation_index, test2_index in kf_validation.split(train[train['Season'] == 1]):
            
        train_index = train_index + len(validation_index) + len(test2_index)
        test1_index = test1_index + len(validation_index) + len(test2_index)
        test_index = np.concatenate([test2_index, test1_index])
            
        x_train, y_train = train.loc[train_index, col_X], train.loc[train_index, 'Yards']
        x_validation, y_validation = train.loc[validation_index, col_X], train.loc[validation_index, 'Yards']
        x_test, y_test = train.loc[test_index, col_X], train.loc[test_index, 'Yards']
        
        lgb_train = lightgbm.Dataset(x_train.values, y_train.values.ravel())
        lgb_validation = lightgbm.Dataset(x_validation.values, y_validation.values.ravel())
        
        evals_results={}
        print('Start training ...')
        model = lightgbm.train(params=param_classifieur , train_set=lgb_train,
                               valid_sets=[lgb_train, lgb_validation], valid_names=['Train', 'Validation'],
                               feature_name = col_X,
                               feval=evaluationFunction,
                               evals_result=evals_results,
                               num_boost_round=1000,
                               early_stopping_rounds=10)
        break

print('... End Training')

      
################################################################################################################################
######################################################## SUBMISSION ############################################################
################################################################################################################################
      
print('Start prediction ...')


for (test_df, sample_sub) in env.iter_test():
    ind = [test_df['PlayId'].values[0]]
    preprocessing(df = test_df)
    convertObjectFeatures(df = test_df)
    test_df = createPlayersPositionTest(df = test_df, features = features_players)
    featuresEngineering(test_df)
    prediction_test = pd.DataFrame(model.predict(test_df[col_X].values)) #keep the same order of the columns like training
    prediction_test = prediction_test.cumsum(axis=1)
    prediction_test.columns = ['Yards'+str(i) for i in range(-99,100)]
    prediction_test.index = ind
    prediction_test = prediction_test.apply(lambda x: int(1) if x.values>=1. else x.values, axis = 0) #not > 1
    prediction_test.index.name = 'PlayId'
    env.predict(prediction_test)
env.write_submission_file()


print('... End prediction')
