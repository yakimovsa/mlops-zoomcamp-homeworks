import argparse
import pickle
import pandas as pd

def run(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.dropOff_datetime - df.pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(y_pred.mean())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        default="2021",
        help="Trip year."
    )
    parser.add_argument(
        "--month",
        default="3",
        help="Trip month"
    )
    args = parser.parse_args()

    year = int(args.year)
    month = int(args.month)
    run(year, month)
