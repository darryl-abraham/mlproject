import sys
import time
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from functools import reduce
import operator

from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, df, num_feats, cat_feats, selected_cat_feats, selected_num_feats, ordinal_feats, true_binary, false_binary, targets):
        self.df = df
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.selected_cat_feats = selected_cat_feats
        self.selected_num_feats = selected_num_feats
        self.ordinal_feats = ordinal_feats
        self.nominal_feats = [value for value in cat_feats if value not in ordinal_feats]
        self.targets = targets
        self.true_binary = true_binary
        self.false_binary = false_binary

    def preprocess(self, set_type='train'):
        start = time.time()
        print(f"Preprocessing {set_type} set...")
        print(f"Initial shape: {self.df.shape}")
        s = time.time()
        print("Quick cleaning...")
        self.quick_clean()
        e = time.time()
        print(f"Quick cleaning done in {e - s:.2f} seconds.")
        s = time.time()
        print("Encoding binary variables...")
        self.encode_binary()
        e = time.time()
        print(f"Encoding binary variables done in {e - s:.2f} seconds.")
        s = time.time()
        print("Encoding categorical variables...")
        self.encode_categorical()
        e = time.time()
        print(f"Encoding categorical variables done in {e - s:.2f} seconds.")
        if set_type == 'train':
            s = time.time()
            print("Balancing target variable...")
            self.balance_target()
            print("Minimising missing values...")
            self.missing_values()
            e = time.time()
            print(f"Balancing target variable and minimising missing values done in {e - s:.2f} seconds.")
        s = time.time()
        print("Detecting outliers...")
        self.outlier_detection()
        e = time.time()
        print(f"{self.num_outliers} outliers removed in {e - s:.2f} seconds.")
        s = time.time()
        print("Imputing missing values...")
        self.impute()
        e = time.time()
        print(f"Imputing missing values done in {e - s:.2f} seconds.")
        s = time.time()
        print("Feature engineering...")
        self.feature_engineering()
        e = time.time()
        print(f"Feature engineering done in {e - s:.2f} seconds.")
        s = time.time()
        print("Performing PCA...")
        self.pca()
        e = time.time()
        print(f"PCA done in {e - s:.2f} seconds.")
        print("Creating final DataFrame...")
        self.create_final_df()
        print("Final shape: ", self.df_final.shape)
        print("Saving preprocessed DataFrame...")
        self.save_df(set_type=set_type)
        end = time.time()
        print(f"Preprocessing done in {end - start:.2f} seconds.")
    
    def quick_clean(self):
        #self.df.drop(columns='Unnamed: 0', inplace=True)
        self.df['SOLP3'] = self.df['SOLP3'].replace(' ', 99).astype(int)
        self.df['SOLIH'] = self.df['SOLIH'].replace(' ', 99).astype(int)
        self.df['CLUSTER'] = self.df['CLUSTER'].replace(' ', -1).astype(int)
        self.df['ZIP'] = self.df['ZIP'].astype(str).str.replace('-', '').astype(int)

    def encode_binary(self):
        self.df.drop(columns=self.false_binary, inplace=True)
        self.num_feats = [value for value in self.num_feats if value not in self.false_binary]
        self.cat_feats = [value for value in self.cat_feats if value not in self.false_binary]
        self.true_binary = [value for value in self.true_binary if value not in self.false_binary]
        #self.nominal_feats = [value for value in self.nominal_feats if value not in self.false_binary]
        binary_encoding = {}
        for col in self.true_binary:
            binary_encoding[col] = {self.df[col].unique()[0]: 0, self.df[col].unique()[1]: 1}
        for col in self.true_binary:
            encoding = binary_encoding[col]
            self.df[col] = self.df[col].replace(encoding)

    def encode_categorical(self):
        dom_byte_mapping = {'R': 1, 'T': 2, 'S': 3, 'C': 4, 'U': 5}
        rfa_first_byte_mapping = {'F': 1, 'N': 2, 'I': 3, 'L': 5, 'A': 6, 'S': 7}
        rfa_third_byte_mapping = {'A': 1, 'B': 3, 'C': 4, 'D': 8, 'E': 13, 'F': 20, 'G': 25}
        mdmaud_first_byte_mapping = {'D': 1, 'I': 2, 'L': 3, 'C': 4}
        mdmaud_third_byte_mapping = {'L': 1, 'C': 2, 'M': 3, 'T': 4}
        gen_byte_mapping = {np.nan: np.nan, ' ': np.nan, 'U': np.nan, 'J': np.nan, 'M': 0, 'F': 1}
        geo2_byte_mapping = {np.nan: np.nan, ' ': np.nan, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
        chil03_byte_mapping = {np.nan: np.nan, ' ': np.nan, 'B': 3, 'M': 2, 'F': 1}

        # DOMAIN
        self.df['DOMAIN'] = self.df['DOMAIN'].apply(self.byte_encoder, args=(dom_byte_mapping, 2, 0))
        self.df['DOMAIN'] = pd.to_numeric(self.df['DOMAIN'], errors='coerce').astype('Int64')

        # RFA
        for col in self.cat_feats:
            if col.startswith('RFA'):
                if col.endswith('R'):
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(rfa_first_byte_mapping, 1, 0))
                elif col.endswith('A'):
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(rfa_third_byte_mapping, 1, 0))
                elif col.endswith('F'):
                    self.df[col] = self.df[col].replace('X', 0)
                else:
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(rfa_first_byte_mapping, 3, 0)).apply(self.byte_encoder, args=(rfa_third_byte_mapping, 3, 2))
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')

        # MDMAUD
            if col.startswith('MDMAUD'):
                if col.endswith('R'):
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(mdmaud_first_byte_mapping, 1, 0))
                elif col.endswith('A'):
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(mdmaud_third_byte_mapping, 1, 0))
                elif col.endswith('F'):
                    self.df[col] = self.df[col].replace('X', 0)
                else:
                    self.df[col] = self.df[col].str[:-1]
                    self.df[col] = self.df[col].apply(self.byte_encoder, args=(mdmaud_first_byte_mapping, 3, 0)).apply(self.byte_encoder, args=(
                    mdmaud_third_byte_mapping, 3, 2))
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype('Int64')

        # GENDER
        self.df['GENDER'] = self.df['GENDER'].apply(lambda x: gen_byte_mapping.get(x))

        # GEOCODE and GEOCODE2
        self.df['GEOCODE'] = self.df['GEOCODE'].replace(' ', np.nan).astype('Int64')
        self.df['GEOCODE2'] = self.df['GEOCODE2'].apply(lambda x: geo2_byte_mapping.get(x)).astype('Int64')

        # DATASRCE
        self.df['DATASRCE'] = self.df['DATASRCE'].replace(' ', np.nan).astype('Int64')

        # LIFESRC
        self.df['LIFESRC'] = self.df['LIFESRC'].replace(' ', np.nan).astype('Int64')

        # CHILD03
        self.df['CHILD03'] = self.df['CHILD03'].apply(lambda x: chil03_byte_mapping.get(x))

        # ZIP
        self.df['OSOURCE'] = self.df['OSOURCE'].replace(' ', np.nan)
        osource_encoding = {val: idx for idx, val in enumerate(self.df['OSOURCE'].unique())}
        self.df['OSOURCE'] = self.df['OSOURCE'].replace(osource_encoding).astype('Int64')

        # STATE
        self.df['STATE'] = self.df['STATE'].replace(' ', np.nan)
        osource_encoding = {val: idx for idx, val in enumerate(self.df['STATE'].unique())}
        self.df['STATE'] = self.df['STATE'].replace(osource_encoding).astype('Int64')

    def byte_encoder(self, code, byte_mapping, true_len, byte_pos):
        if pd.isna(code) or (isinstance(code, str) and len(code) != true_len):
            return np.nan
        elif code == 0 or 'X' in code:
            return 0
        else:
            byte = byte_mapping.get(code[byte_pos])
            return f"{code[:byte_pos]}{byte}{code[byte_pos + 1:]}"

    def balance_target(self):
        self.df_pos = self.df[self.df['TARGET_B'] == 1]
        self.df_neg = self.df[self.df['TARGET_B'] == 0]

        self.df_neg_sample = self.df_neg.sample(n=len(self.df_pos), random_state=42)
        self.df = pd.concat([self.df_pos, self.df_neg_sample], axis=0)

    def missing_values(self):
        # Drop columns with more than 50% missing values
        threshold = 0.5
        df_ = self.df.dropna(thresh=len(self.df) * threshold, axis=1)
        self.num_feats = [value for value in self.num_feats if value in df_.columns]
        self.cat_feats = [value for value in self.cat_feats if value in df_.columns]
        self.ordinal_feats = [value for value in self.ordinal_feats if value in df_.columns]
        #self.nominal_feats = [value for value in self.nominal_feats if value in df_.columns]
        self.selected_num_feats = [value for value in self.selected_num_feats if value in df_.columns]

        self.df = df_

        """
        # Recursively sample negative class to only allow rows with less than 30% missing values
        self.df_pos = self.df[self.df['TARGET_B'] == 1]
        self.df_neg = self.df[self.df['TARGET_B'] == 0]
        df_neg_sample = self.recursive_sample(self.df_neg, len(self.df_pos), 0.7)
        self.df = pd.concat([self.df_pos, df_neg_sample], axis=0)
        """

        # get negative rows with the least missing values
        self.df_pos = self.df[self.df['TARGET_B'] == 1]
        df_neg_sample = self.df.loc[self.df.isnull().sum(axis=1).nsmallest(len(self.df_pos)).index]
        self.df = pd.concat([self.df_pos, df_neg_sample], axis=0)

    def recursive_sample(self, df, n, threshold):
        df_neg = self.df[self.df['TARGET_B'] == 0]
        sample = df_neg.sample(n)
        temp_ = sample.dropna(thresh=len(df.columns) * threshold, axis=0)
        if n - len(temp_) == 0:
            return sample
        else:
            return self.recursive_sample(df, n, threshold)

    def outlier_detection(self, factor=3):
        Q1 = self.df[self.num_feats].quantile(0.25)
        Q3 = self.df[self.num_feats].quantile(0.75)
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        # Number of outliers
        self.num_outliers = len(self.df[(self.df[self.num_feats] < lower_bound) | (self.df[self.num_feats] > upper_bound)])
        # Replace outliers with np.nan
        self.df[self.num_feats] = self.df[self.num_feats].mask((self.df[self.num_feats] < lower_bound) | (self.df[self.num_feats] > upper_bound))

    def impute(self):
        # numerical feature imputation
        nimputer = KNNImputer(n_neighbors=10)
        self.df[self.selected_num_feats] = nimputer.fit_transform(self.df[self.selected_num_feats])

        # ordinal feature imputation
        cimputer = KNN(k=10, verbose=False)
        self.df[self.ordinal_feats] = cimputer.fit_transform(self.df[self.ordinal_feats]).round()

        # nominal feature imputation
        self.nominal_feats = [value for value in self.cat_feats if value not in self.ordinal_feats]
        self.df[self.nominal_feats] = self.df[self.nominal_feats].apply(lambda x: x.fillna(x.mode()[0]))

    def feature_engineering(self):
        # Merge DOMAIN subvariables
        self.df['DOMAIN'] = self.df['DOMAIN'].apply(self.merge_subvars)

        # Merge RFA subvariables
        for col in self.cat_feats:
            if col.startswith('RFA'):
                self.df[col] = self.df[col].apply(self.merge_subvars)

        # Merge MDMAUD subvariables
            if col.startswith('MDMAUD'):
                self.df[col] = self.df[col].apply(self.merge_subvars)

    def merge_subvars(self, code):
        if pd.isna(code):
            return np.nan
        elif isinstance(code, str):
            return code
        else:
            bytes = [int(byte) for byte in str(int(code))]
            new = reduce(operator.mul, bytes)
            return new

    def pca(self, n_components=13):
        # Numerical variable preprocessing: Standardization
        scaler = StandardScaler()
        scaled_numerical_data = scaler.fit_transform(self.df[self.selected_num_feats])

        # Perform PCA for numerical variables
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_numerical_data)

        # Create DataFrame to store PCA results
        self.pca_df = pd.DataFrame(pca_result, columns=[f'PC{i}' for i in range(1, n_components + 1)],
                              index=self.df[self.selected_num_feats].index)

        # Get explained variance ratio
        self.pca_exp_var_ratio = pca.explained_variance_ratio_

    def create_final_df(self):
        #self.df_final = self.df[self.selected_cat_feats].join(self.pca_df.iloc[:, :72], how='inner')
        self.df_final = pd.concat([self.df[self.selected_cat_feats], self.pca_df.iloc[:, :13], self.df[self.targets]], join='inner', axis=1, ignore_index=False)

    def save_df(self, set_type='train'):
        if set_type == 'train':
            self.df_final.to_csv('data/preprocessed_data.csv', index=True)
        elif set_type == 'test':
            self.df_final.to_csv('data/test_preprocessed.csv', index=True)


if __name__ == '__main__':
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    df = pd.read_csv(arg2, sep=',', low_memory=False)

    id = ['CONTROLN']
    num_feats = ['ODATEDW', 'DOB', 'AGE', 'NUMCHLD', 'INCOME', 'HIT', 'MBCRAFT', "MBGARDEN", "MBBOOKS", "MBCOLECT",
                 "MAGFAML", "MAGFEM", "MAGMALE", "PUBGARDN", "PUBCULIN", "PUBHLTH", "PUBDOITY", "PUBNEWFN", "PUBPHOTO",
                 "PUBOPP", 'MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 'STATEGOV', 'FEDGOV', 'POP901',
                 'POP902', 'POP903', 'POP90C1', 'POP90C2', 'POP90C3', 'POP90C4', 'POP90C5', 'ETH1', 'ETH2', 'ETH3',
                 'ETH4', 'ETH5', 'ETH6', 'ETH7', 'ETH8', 'ETH9', 'ETH10', 'ETH11', 'ETH12', 'ETH13', 'ETH14', 'ETH15',
                 'ETH16', 'AGE901', 'AGE902', 'AGE903', 'AGE904', 'AGE905', 'AGE906', 'AGE907', 'CHIL1', 'CHIL2',
                 'CHIL3', 'AGEC1', 'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5', 'AGEC6', 'AGEC7', 'CHILC1', 'CHILC2', 'CHILC3',
                 'CHILC4', 'CHILC5', "HHAGE1", "HHAGE2", "HHAGE3", "HHN1", "HHN2", "HHN3", "HHN4", "HHN5", "HHN6",
                 "MARR1", "MARR2", "MARR3", "MARR4", "HHP1", "HHP2", "DW1", "DW2", "DW3", "DW4", "DW5", "DW6", "DW7",
                 "DW8", "DW9", "HV1", "HV2", "HV3", "HV4", "HU1", "HU2", "HU3", "HU4", "HU5", "HHD1", "HHD2", "HHD3",
                 "HHD4", "HHD5", "HHD6", "HHD7", "HHD8", "HHD9", "HHD10", "HHD11", "HHD12", "ETHC1", "ETHC2", "ETHC3",
                 "ETHC4", "ETHC5", "ETHC6", "HVP1", "HVP2", "HVP3", "HVP4", "HVP5", "HVP6", "HUR1", "HUR2", "RHP1",
                 "RHP2", "RHP3", "RHP4", "HUPA1", "HUPA2", "HUPA3", "HUPA4", "HUPA5", "HUPA6", "HUPA7", "RP1", "RP2",
                 "RP3", "RP4", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "IC10", "IC11", "IC12",
                 "IC13", "IC14", "IC15", "IC16", "IC17", "IC18", "IC19", "IC20", "IC21", "IC22", "IC23", "HHAS1",
                 "HHAS2", "HHAS3", "HHAS4", "MC1", "MC2", "MC3", "TPE1", "TPE2", "TPE3", "TPE4", "TPE5", "TPE6", "TPE7",
                 "TPE8", "TPE9", "PEC1", "PEC2", "TPE10", "TPE11", "TPE12", "TPE13", "LFC1", "LFC2", "LFC3", "LFC4",
                 "LFC5", "LFC6", "LFC7", "LFC8", "LFC9", "LFC10", "OCC1", "OCC2", "OCC3", "OCC4", "OCC5", "OCC6",
                 "OCC7", "OCC8", "OCC9", "OCC10", "OCC11", "OCC12", "OCC13", "EIC1", "EIC2", "EIC3", "EIC4", "EIC5",
                 "EIC6", "EIC7", "EIC8", "EIC9", "EIC10", "EIC11", "EIC12", "EIC13", "EIC14", "EIC15", "EIC16", "OEDC1",
                 "OEDC2", "OEDC3", "OEDC4", "OEDC5", "OEDC6", "OEDC7", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7",
                 "EC8", "SEC1", "SEC2", "SEC3", "SEC4", "SEC5", "AFC1", "AFC2", "AFC3", "AFC4", "AFC5", "AFC6", "VC1",
                 "VC2", "VC3", "VC4", "ANC1", "ANC2", "ANC3", "ANC4", "ANC5", "ANC6", "ANC7", "ANC8", "ANC9", "ANC10",
                 "ANC11", "ANC12", "ANC13", "ANC14", "ANC15", "POBC1", "POBC2", "LSC1", "LSC2", "LSC3", "LSC4", "VOC1",
                 "VOC2", "VOC3", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HC8", "HC9", "HC10", "HC11", "HC12",
                 "HC13", "HC14", "HC15", "HC16", "HC17", "HC18", "HC19", "HC20", "HC21", "MHUC1", "MHUC2", "AC1", "AC2",
                 'CARDPROM', 'NUMPROM', 'CARDPM12', 'NUMPRM12', "RAMNT_3", "RAMNT_4", "RAMNT_5", "RAMNT_6", "RAMNT_7",
                 "RAMNT_8", "RAMNT_9", "RAMNT_10", "RAMNT_11", "RAMNT_12", "RAMNT_13", "RAMNT_14", "RAMNT_15",
                 "RAMNT_16", "RAMNT_17", "RAMNT_18", "RAMNT_19", "RAMNT_20", "RAMNT_21", "RAMNT_22", "RAMNT_23",
                 "RAMNT_24", "RAMNTALL", "NGIFTALL", "CARDGIFT", "MINRAMNT", "MINRDATE", "MAXRAMNT", "MAXRDATE",
                 "LASTGIFT", "LASTDATE", "FISTDATE", "NEXTDATE", "TIMELAG", "AVGGIFT", "ADATE_2", "ADATE_3", "ADATE_4",
                 "ADATE_5", "ADATE_6", "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10", "ADATE_11", "ADATE_12", "ADATE_13",
                 "ADATE_14", "ADATE_15", "ADATE_16", "ADATE_17", "ADATE_18", "ADATE_19", "ADATE_20", "ADATE_21",
                 "ADATE_22", "ADATE_23", "ADATE_24"]
    cat_feats = ['OSOURCE', 'TCODE', 'STATE', 'ZIP', 'MAILCODE', 'PVASTATE', 'NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG',
                 'RECSWEEP', 'MDMAUD', 'DOMAIN', 'CLUSTER', 'AGEFLAG', 'HOMEOWNR', 'CHILD03', 'CHILD07', 'CHILD12',
                 'CHILD18', 'GENDER', 'WEALTH1', 'dfSRCE', 'SOLP3', 'SOLIH', 'MAJOR', 'WEALTH2', 'GEOCODE',
                 'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO',
                 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES', 'LIFESRC',
                 'PEPSTRFL', "MSA", "ADI", "DMA", "RFA_2", "RFA_3", "RFA_4", "RFA_5", "RFA_6", "RFA_7", "RFA_8",
                 "RFA_9", "RFA_10", "RFA_11", "RFA_12", "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17", "RFA_18",
                 "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23", "RFA_24", 'MAXADATE', 'HPHONE_D', "RFA_2R", "RFA_2F",
                 "RFA_2A", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A", 'CLUSTER2', 'GEOCODE2']
    selected_cat_feats = ['OSOURCE', 'TCODE', 'STATE', 'MAILCODE', 'RECINHSE', 'RECP3', 'CLUSTER', 'WEALTH1', 'SOLIH',
                          'WEALTH2', 'VETERANS', 'CRAFTS', 'WALKER', 'PEPSTRFL', 'RFA_2', 'RFA_3', 'RFA_4', 'RFA_5',
                          'RFA_6', 'RFA_7', 'RFA_8', 'RFA_9', 'RFA_10', 'RFA_11', 'RFA_12', 'RFA_13', 'RFA_14',
                          'RFA_16', 'RFA_17', 'RFA_18', 'RFA_19', 'RFA_21', 'RFA_22', 'RFA_24', 'RFA_2F', 'RFA_2A',
                          'CLUSTER2']
    selected_num_feats = ['INCOME', 'MAGMALE', 'POP90C1', 'POP90C2', 'POP90C3', 'ETH16', 'MC1', 'MC2', 'OCC4', 'OCC6',
                          'OCC7', 'OCC9', 'OCC10', 'OCC11', 'OCC12', 'EC1', 'VOC2', 'VOC3', 'HC2', 'MHUC2', 'CARDPROM',
                          'CARDPM12', 'RAMNT_8', 'RAMNT_9', 'RAMNT_14', 'RAMNT_16', 'RAMNT_19', 'RAMNT_21', 'RAMNT_22',
                          'RAMNT_23', 'RAMNTALL', 'NGIFTALL', 'CARDGIFT', 'MINRDATE', 'LASTGIFT', 'LASTDATE',
                          'AVGGIFT', 'ADATE_5', 'ADATE_12', 'ADATE_15', 'ADATE_20']
    ordinal_feats = ['DOMAIN', 'WEALTH1', 'SOLP3', 'SOLIH', 'WEALTH2', "RFA_2", "RFA_3", "RFA_4", "RFA_5", "RFA_6",
                     "RFA_7", "RFA_8", "RFA_9", "RFA_10", "RFA_11", "RFA_12", "RFA_13", "RFA_14", "RFA_15", "RFA_16",
                     "RFA_17", "RFA_18", "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23", "RFA_24", "RFA_2R", "RFA_2F",
                     "RFA_2A", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A"]
    # Binary by definition
    true_binary = [
        'MAILCODE', 'PVASTATE', 'NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'AGEFLAG', 'HOMEOWNR', 'CHILD07',
        'CHILD12', 'CHILD18', 'MAJOR', 'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY', 'STEREO',
        'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES',
        'PEPSTRFL', 'TARGET_B', 'HPHONE_D'
    ]
    # Not binary in dataset
    false_binary = ['HOMEOWNR', 'NOEXCH', 'CHILD18', 'CHILD12', 'AGEFLAG', 'PVASTATE', 'CHILD07']
    targets = ['TARGET_B', 'TARGET_D']

    preprocessor = Preprocessor(df, num_feats, cat_feats, selected_cat_feats, selected_num_feats, ordinal_feats, true_binary, false_binary, targets)
    preprocessor.preprocess(set_type=arg1)

