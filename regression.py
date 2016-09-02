#!/usr/bin/env python

"""Takes list of predictors from a greedy algorithm output (from Oskar's code)
and uses the existing dataset to generate an R^2 value that characterizes the fit of the
Oskar model. To run, use the following command:

python regression.py <extended-regression-data-matrix-with-split-dates.csv --variables 
file_with_predictors_list.csv --year 1994

Can use any year for the test/training split"""

import pandas as pd
import statsmodels.formula.api as smf
import sys
import argparse
from sklearn.metrics import r2_score
from greedy import collinify

parser = argparse.ArgumentParser(
    description=('xxx...'))

parser.add_argument(
    '--variables', required=True,
    help=('The filename that contains the variables to use.'))

parser.add_argument(
    '--year', required=True, type=int,
    help=('Give the year for the cut-off between training and '
          'test data.'))

args = parser.parse_args()

variables = open(args.variables).read().split()
# print('variables are:', variables)

data = pd.read_csv(sys.stdin, sep=' ')

data = collinify(data)

training = data.loc[(data['YEAR1'] <= args.year) &
                    (data['YEAR2'] <= args.year)]

results = smf.ols(formula='AGDIST ~ %s' % ' + '.join(variables),
                  data=training).fit()

# print('Summary of OLS fit:')
# print(results.summary())

test = data.loc[(data['YEAR1'] > args.year) &
                (data['YEAR2'] > args.year)]

predicted = results.predict(test)
actual = test['AGDIST']

r2 = r2_score(actual, predicted)

print(r2)
