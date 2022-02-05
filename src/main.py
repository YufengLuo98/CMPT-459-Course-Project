import helper1 # include helper functions
import numpy as np
import pandas as pd

# Load all datasets
cases_train = pd.read_csv('./datasets/cases_2021_train.csv')
cases_test = pd.read_csv('./datasets/cases_2021_test.csv')
cases_location = pd.read_csv('./datasets/location_2021.csv')


# 1.1 Cleaning messy outcome labels
cases_train.groupby('outcome').size()

labels = {
    'Discharged': 'hospitalized', 
    'Discharged from hospital': 'hospitalized', 
    'Hospitalized': 'hospitalized', 
    'critical condition': 'hospitalized', 
    'discharge': 'hospitalized', 
    'discharged': 'hospitalized', 
    'Alive': 'nonhospitalized', 
    'Receiving Treatment': 'nonhospitalized', 
    'Stable': 'nonhospitalized', 
    'Under treatment': 'nonhospitalized', 
    'recovering at home 03.03.2020': 'nonhospitalized', 
    'released from quarantine': 'nonhospitalized', 
    'stable': 'nonhospitalized', 
    'stable condition': 'nonhospitalized', 
    'Dead': 'deceased', 
    'Death': 'deceased', 
    'Deceased': 'deceased', 
    'Died': 'deceased', 
    'death': 'deceased', 
    'died': 'deceased',
    'Recovered': 'recovered', 
    'recovered': 'recovered'
}

# Mapping similar outcomes
cases_train['outcome_group'] = cases_train['outcome'].map(labels)
cases_train = cases_train.drop(columns=['outcome'])


# 1.4 Data Cleaning and Imputing Missing Values
# remove rows with empty age attribute
cases_train = cases_train[cases_train['age'].notna()].reset_index()
cases_test = cases_test[cases_test['age'].notna()].reset_index()

# Strip all whitespace from 'age' columns
cases_train['age'] = cases_train['age'].str.strip()
cases_test['age'] = cases_test['age'].str.strip()

cases_train = cases_train.drop(columns=['index'])
cases_test = cases_test.drop(columns=['index'])

# Cleaning ranges in age attribute
cases_train['age'] = cases_train['age'].apply(lambda x: helper1.range_to_num(x))
cases_train = cases_train[cases_train['age'] != 'remove']
cases_train['age'] = pd.to_numeric(cases_train['age'])
cases_train['age'] = cases_train['age'].apply(lambda x: round(x))

cases_test['age'] = cases_test['age'].apply(lambda x: helper1.range_to_num(x))
cases_test = cases_test[cases_test['age'] != 'remove']
cases_test['age'] = pd.to_numeric(cases_test['age'])
cases_test['age'] = cases_test['age'].apply(lambda x: round(x))

# get missing country information
cases_train['country'] = cases_train.apply(lambda x: helper1.get_country(cases_location,
			  x.latitude, x.longitude) if pd.isnull(x['country'])
			  else x.country, axis=1)
cases_test['country'] = cases_test.apply(lambda x: helper1.get_country(cases_location,
			 x.latitude, x.longitude) if pd.isnull(x['country']) 
			 else x.country, axis=1)

# replace nan with 'unknown' for sex, date_confirmation, additional_information and source
cases_train[['sex', 'date_confirmation', 'additional_information', 'source']] = cases_train[['sex', 'date_confirmation', 'additional_information', 'source']].fillna('unknown')
cases_test[['sex', 'date_confirmation', 'additional_information', 'source']] = cases_test[['sex', 'date_confirmation', 'additional_information', 'source']].fillna('unknown')

# Clean the locations data
cases_location.dropna(subset=['Incident_Rate', 'Case_Fatality_Ratio'], inplace=True)


# 1.5 Dealing With Outliers
# set binary to True if chronic disease is mentioned in addional information
cases_test['chronic_disease_binary'] = cases_test.apply(lambda x: True if
 					pd.notnull(x.additional_information) and 
 					"chronic" in x.additional_information.lower()
 					else x.chronic_disease_binary, axis=1)

cases_train['chronic_disease_binary'] = cases_train.apply(lambda x: True if
					 pd.notnull(x.additional_information) and 
					 "chronic" in x.additional_information.lower() 
					 else x.chronic_disease_binary, axis=1)

# Remove locations that have confirmed cases > 1000 and case_fatality_ratio < 0.1
cases_location = cases_location[~((cases_location['Confirmed'] > 1000) & (cases_location['Case_Fatality_Ratio'] < 0.1))].reset_index(drop=True)


# 1.6 Joining the cases and location dataset
cases_location['Country_Region'] = cases_location['Country_Region'].apply(helper1.fix_country_label)

# Group the location data by province and country
cases_location['Province_State'].fillna('', inplace=True)
cases_location['Population'] = cases_location['Confirmed'] * 100000 /	 					cases_location['Incident_Rate']
				
grouped_locations = cases_location.groupby(['Province_State', 'Country_Region']).agg({'Lat': 'mean', 
					   	 					'Long_': 'mean',
					   	 					'Confirmed': 'sum',
					   	 					'Deaths': 'sum',
					   	 					'Recovered': 'sum',
					   	 					'Active': 'sum',
					   	 					'Population': 'sum'}).reset_index()
					   	 					
# recalculate ratios for grouped locations				   	 
grouped_locations['Incident_Rate'] = grouped_locations['Confirmed'] /      					      grouped_locations['Population'] * 100000 
grouped_locations['Case_Fatality_Ratio'] = grouped_locations['Deaths'] /   						    grouped_locations['Confirmed'] * 100 

# adds province / country combination to set
location_set = set()
def add_to_location_set(row):
    location_set.add((row['Province_State'], row['Country_Region']))
    
grouped_locations.apply(add_to_location_set, axis=1)

# get nearest provice
cases_train['province'] = cases_train.apply(lambda x: helper1.get_province(cases_location, 
					     x.country, x.latitude, x.longitude) if pd.isnull(x['province']) 
					     or (x.province, x.country) not in location_set 
					     else x.province, axis=1)
cases_test['province'] = cases_test.apply(lambda x: helper1.get_province(cases_location, 
					   x.country, x.latitude, x.longitude) if pd.isnull(x['province'])
					   or (x.province, x.country) not in location_set
					   else x.province, axis=1)

# join datasets
combined_train = pd.merge(cases_train, grouped_locations, how='inner', left_on=['country', 'province'], right_on = ['Country_Region', 'Province_State'])
combined_test = pd.merge(cases_test, grouped_locations, how='inner', left_on=['country', 'province'], right_on = ['Country_Region', 'Province_State'])

combined_train.columns = combined_train.columns.str.lower()
combined_test.columns = combined_train.columns.str.lower()

# Save processed data to results directory
combined_train.to_csv('results/cases_2021_train_processed.csv', index=False)
combined_test.to_csv('results/cases_2021_test_processed.csv', index=False)
grouped_locations.to_csv('results/location_2021_processed.csv', index=False)


# 1.7 Feature Selection
# Extract the features from the combined data
train_features = combined_train[['age', 'sex', 'province', 'country', 'date_confirmation', 'chronic_disease_binary', 'confirmed', 'deaths', 'population', 'incident_rate', 'case_fatality_ratio', 'outcome_group']]
test_features = combined_test[['age', 'sex', 'province', 'country', 'date_confirmation', 'chronic_disease_binary', 'confirmed', 'deaths', 'population', 'incident_rate', 'case_fatality_ratio', 'outcome_group']]

# Save extracted features to results directory
train_features.to_csv('results/cases_2021_train_processed_features.csv', index=False)
test_features.to_csv('results/cases_2021_test_processed_features.csv', index=False)
