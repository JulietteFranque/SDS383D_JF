import pandas as pd

def make_fips_df(file_path='../raw_data/census_counts.csv'):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'address.location.social_vulnerability_2016.general.fips': 'full_fips_code'})
    df['department_name'] = df['department_name'].str.replace(r" \(.*\)", '', regex=True)
    df['county_fips_code'] = df['full_fips_code'].apply(lambda x: str(x)[0:5]).astype(int)
    grouped_df = \
        df[['department_name', 'county_fips_code', 'incident_count']].groupby(['department_name', 'county_fips_code'])[
            'incident_count'].sum().to_frame().reset_index()
    fips_per_department = \
        grouped_df.loc[grouped_df.groupby(['department_name'])['incident_count'].idxmax()].reset_index(drop=True)[
            ['department_name', 'county_fips_code']]
    fips_per_department.to_csv('../formatted_data/county_fips_per_department.csv', index=False)
    return fips_per_department

