import pickle
import argparse

#Workflow to query from the command line:

# Once (or after data changes)
#python fit_models.py

# Anytime after
#python query_OLS_models.py --label "SPY_flow ~ ASVI4" --year 2022
#python query_OLS_models.py --label "mag7_ret ~ ASVI8"
#python query_OLS_models.py --year 2020

with open("models.pkl", "rb") as f:
    models = pickle.load(f)

valid_labels = list(models.keys())
valid_years  = list(next(iter(models.values())).keys())

parser = argparse.ArgumentParser()
parser.add_argument("--label", choices=valid_labels)
parser.add_argument("--year",  choices=valid_years)
args = parser.parse_args()

if args.label and args.year:
    print(models[args.label][args.year].summary())

elif args.label:
    for year, result in models[args.label].items():
        print(f"\n=== {args.label} | {year} ===")
        print(result.summary())

elif args.year:
    for label, year_results in models.items():
        print(f"\n=== {label} | {args.year} ===")
        print(year_results[args.year].summary())

else:
    parser.print_help()

