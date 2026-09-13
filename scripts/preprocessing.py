"""Step 1: EPL inspection, cleaning, prematch features and transformations."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULT_ENCODING = {'A': 0, 'D': 1, 'H': 2}
IDS = ['Season', 'MatchDate', 'HomeTeam', 'AwayTeam']
COUNTS = ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HalfTimeHomeGoals', 'HalfTimeAwayGoals',
          'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
          'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
          'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards']
SHOTS = ['HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget']
REPORT = ROOT / 'reports/step1'


def load_source():
    data = pd.read_csv(ROOT / 'datasets/epl_source_normalized.csv')
    data['MatchDate'] = pd.to_datetime(data['MatchDate'], format='%Y-%m-%d', errors='raise')
    for col in COUNTS:
        data[col] = pd.to_numeric(data[col], errors='raise')
    return data.sort_values(['MatchDate', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)


def inspect_data(data):
    REPORT.mkdir(parents=True, exist_ok=True)
    data[COUNTS].agg(['count', 'mean', 'median', 'min', 'max', 'std']).T.to_csv(REPORT / 'numeric_statistics.csv')
    pd.DataFrame({'dtype': data.dtypes.astype(str), 'missing': data.isna().sum()}).to_csv(REPORT / 'column_inspection.csv')
    print(f'Shape: {data.shape[0]:,} rows × {data.shape[1]} columns')
    print(f'Dates: {data.MatchDate.min().date()} to {data.MatchDate.max().date()}')
    print('Target: FullTimeResult (A=away win, D=draw, H=home win)')
    print(data.FullTimeResult.value_counts().to_string())
    print('\nNumeric statistics:')
    print(data[COUNTS].agg(['mean', 'median', 'min', 'max']).T.round(3).to_string())
    print('\nTypes and missing values:')
    print(pd.DataFrame({'dtype': data.dtypes.astype(str), 'missing': data.isna().sum()}).to_string())


def clean_data(data):
    data = data.copy()
    keys = ['MatchDate', 'HomeTeam', 'AwayTeam']
    # Drop exact duplicates only; conflicting records require investigation.
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    if data.duplicated(keys).any():
        raise ValueError('Conflicting duplicate fixtures require source review')
    if data[IDS + ['FullTimeResult']].isna().any().any():
        raise ValueError('Missing fixture identity/target requires source review')
    if not data.FullTimeResult.isin(RESULT_ENCODING).all():
        raise ValueError('Unexpected result label')
    if (data[COUNTS] < 0).any().any():
        raise ValueError('Negative match counts require source review')
    invalid_sides = 0
    for side in ('Home', 'Away'):
        a, b = side + 'Shots', side + 'ShotsOnTarget'
        invalid = data[b] > data[a]
        invalid_sides += int(invalid.sum())
        data.loc[invalid, [a, b]] = np.nan
    # Descriptive match data: impute only unreliable shooting observations.
    # Each date uses the median of valid provider values on strictly earlier dates.
    # Keep flags so predictive histories can ignore estimated observations.
    for col in SHOTS:
        data[col + 'Imputed'] = data[col].isna().astype('int8')
        valid_history = []
        for _, group in data.groupby('MatchDate', sort=True):
            observed = group[col].dropna().tolist()
            missing_index = group.index[group[col].isna()]
            if len(missing_index):
                if not valid_history:
                    raise ValueError(f'No earlier observations available to impute {col}')
                data.loc[missing_index, col] = float(np.median(valid_history))
            valid_history.extend(observed)
    data['ResultEncoded'] = data.FullTimeResult.map(RESULT_ENCODING).astype('int8')
    data['HalfTimeResultEncoded'] = data.HalfTimeResult.map(RESULT_ENCODING).astype('int8')
    print(f'Exact duplicates removed: {exact_duplicates}')
    print(f'Invalid shooting team-sides: {invalid_sides}; imputed cells: {int(data[[c + "Imputed" for c in SHOTS]].sum().sum())}')
    print('Missing values after cleaning:', int(data.isna().sum().sum()))
    print('Result label encoding:', RESULT_ENCODING)
    print('Legitimate extreme scores retained. No unsupported rare-team exclusions.')
    return data


def team_history(clean):
    pieces = []
    for side, other in [('Home', 'Away'), ('Away', 'Home')]:
        h = pd.DataFrame({'Date': clean.MatchDate, 'Team': clean[side + 'Team'],
                          'Venue': side, 'GoalsFor': clean['FullTime' + side + 'Goals'],
                          'GoalsAgainst': clean['FullTime' + other + 'Goals']})
        h['Points'] = np.where(h.GoalsFor > h.GoalsAgainst, 3, np.where(h.GoalsFor == h.GoalsAgainst, 1, 0))
        h['GoalDifference'] = h.GoalsFor - h.GoalsAgainst
        for metric in ('Shots', 'ShotsOnTarget'):
            h[metric] = clean[side + metric].where(clean[side + metric + 'Imputed'] == 0)
        pieces.append(h)
    return pd.concat(pieces, ignore_index=True).sort_values('Date').reset_index(drop=True)


def summarize(history, team, venue, when):
    eligible = history[(history.Team == team) & (history.Date < when) &
                       (history.Date >= when - pd.Timedelta(days=365))]
    recent = eligible.tail(5)
    specific = eligible[eligible.Venue == venue].tail(5)
    result = {metric + 'Last5': recent[metric].mean() for metric in
              ['Points', 'GoalsFor', 'GoalsAgainst', 'GoalDifference', 'Shots', 'ShotsOnTarget']}
    result.update({'Venue' + metric + 'Last5': specific[metric].mean()
                   for metric in ['Points', 'GoalsFor', 'GoalsAgainst']})
    result['DaysSinceEPLMatch'] = (when - eligible.Date.iloc[-1]).days if len(eligible) else np.nan
    result['HistoryCount'] = len(recent)
    result['VenueHistoryCount'] = len(specific)
    result['ShotsObservationCount'] = recent.Shots.count()
    result['ShotsOnTargetObservationCount'] = recent.ShotsOnTarget.count()
    return result


def fixture_features(history, home, away, when):
    """Same entry point for training and future UI predictions."""
    if home == away:
        raise ValueError('Home and away teams must differ')
    when = pd.Timestamp(when).normalize()
    h = summarize(history, home, 'Home', when)
    a = summarize(history, away, 'Away', when)
    return {**{'Home' + k: v for k, v in h.items()},
            **{'Away' + k: v for k, v in a.items()},
            **{'Difference' + k: h[k] - a[k] for k in h}}


def engineer_features(clean):
    history = team_history(clean)
    # Filter to the two teams before each call to avoid scanning all team strings.
    teams = {team: frame for team, frame in history.groupby('Team')}
    values = []
    for row in clean.itertuples(index=False):
        pair = pd.concat([teams[row.HomeTeam], teams[row.AwayTeam]]).sort_values('Date')
        values.append(fixture_features(pair, row.HomeTeam, row.AwayTeam, row.MatchDate))
    result = pd.concat([clean[IDS + ['FullTimeResult', 'ResultEncoded']].reset_index(drop=True),
                        pd.DataFrame(values)], axis=1)
    print(f'Engineered {len(values):,} rows with {len(values[0])} prematch numeric features.')
    print('Only strictly earlier dates within 365 days contribute to each last-five window.')
    return result


def make_transformer():
    # Refit this entire pipeline inside every Step 3/4 training fold.
    return Pipeline([('median_imputation', SimpleImputer(strategy='median', add_indicator=True,
                                                        keep_empty_features=True)),
                     ('standardization', StandardScaler())])


def demonstrate_transformations(features):
    feature_cols = [c for c in features if c not in IDS + ['FullTimeResult', 'ResultEncoded']]
    # Reference fit solely to verify Step 1 transformations; no prediction model is fitted.
    # 2000/01 supplies warm-up history; 2026/27 is partial and excluded.
    eligible = features[(features.Season != '2000/01') & (features.Season != '2026/27')]
    n_train = int(0.8 * len(eligible))
    training, testing = eligible.iloc[:n_train], eligible.iloc[n_train:]
    assert training.MatchDate.max() < testing.MatchDate.min()
    transformer = make_transformer()
    transformed_train = transformer.fit_transform(training[feature_cols])
    transformed_test = transformer.transform(testing[feature_cols])
    assert np.isfinite(transformed_train).all() and np.isfinite(transformed_test).all()
    variable = np.std(transformed_train, axis=0) > 1e-10
    assert np.allclose(transformed_train.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(transformed_train.std(axis=0)[variable], 1, atol=1e-10)
    result = {'purpose': 'Step 1 transformation verification only; refit within Step 3/4 folds',
              'training_rows': len(training), 'testing_rows': len(testing),
              'training_last_date': str(training.MatchDate.max().date()),
              'testing_first_date': str(testing.MatchDate.min().date()),
              'input_feature_count': len(feature_cols), 'transformed_feature_count': transformed_train.shape[1],
              'training_missing_before_imputation': int(training[feature_cols].isna().sum().sum()),
              'testing_missing_before_imputation': int(testing[feature_cols].isna().sum().sum()),
              'missing_after_transformation': 0,
              'training_medians': dict(zip(feature_cols, transformer.named_steps['median_imputation'].statistics_.tolist()))}
    (REPORT / 'transformation_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Transformation reference: {len(training):,} earlier training rows / {len(testing):,} later test rows (80/20).')
    print(f'{len(feature_cols)} inputs become {transformed_train.shape[1]} columns including missingness indicators.')
    print('Median imputation and standardization verified: finite output; training means 0, nonconstant standard deviations 1.')
    print('Unfitted feature CSV is preserved so later folds can fit their own medians and scaling.')
    return result


def create_charts(before, after):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    invalid = [int((before[s + 'ShotsOnTarget'] > before[s + 'Shots']).sum()) for s in ('Home', 'Away')]
    x = np.arange(2)
    ax.bar(x - .18, invalid, .36, label='Invalid provider observations', color='#b44d43')
    ax.bar(x + .18, [int(after[s+'ShotsImputed'].sum()) for s in ('Home', 'Away')], .36,
           label='Flagged and median-imputed', color='#277a83')
    ax.set(xticks=x, xticklabels=['Home team', 'Away team'], ylabel='Team-side observations',
           title='Shooting data: invalid observations and treatment', ylim=(0, 4))
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(REPORT / 'shooting_data_treatment.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained', sharey=True)
    for ax, frame, title in zip(axes, [before, after], ['Before cleaning', 'After cleaning']):
        totals = frame.FullTimeHomeGoals + frame.FullTimeAwayGoals
        ax.hist(totals, bins=np.arange(-.5, totals.max()+1.5), color='#277a83', edgecolor='white')
        ax.set(title=title, xlabel='Total full-time goals')
    axes[0].set_ylabel('Matches')
    fig.suptitle('Valid match outcomes are retained during cleaning')
    fig.savefig(REPORT / 'goals_before_after.png', dpi=160)
    plt.close(fig)
    print('Saved two Chapter 2.3 before/after preprocessing figures.')


def export_data(clean, features):
    clean.to_csv(ROOT / 'Cleaned_Dataset.csv', index=False, date_format='%Y-%m-%d')
    features.to_csv(ROOT / 'datasets/Prematch_Features.csv', index=False, date_format='%Y-%m-%d')
    print('Exported Cleaned_Dataset.csv: canonical match data for ALL downstream notebooks.')
    print('Exported datasets/Prematch_Features.csv: derived features; missing history is imputed inside training folds.')


def run():
    data = load_source()
    inspect_data(data)
    clean = clean_data(data)
    features = engineer_features(clean)
    demonstrate_transformations(features)
    create_charts(data, clean)
    export_data(clean, features)


if __name__ == '__main__':
    run()
