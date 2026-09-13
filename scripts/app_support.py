"""Shared data validation and forecast logic used by the Streamlit UI."""
import pandas as pd
import numpy as np
from .preprocessing import fixture_features, team_history


def predict_fixture(clean,bundle,model_name,home,away,when):
    if home==away:
        raise ValueError('Choose two different teams.')
    teams=set(clean.HomeTeam)|set(clean.AwayTeam)
    if home not in teams or away not in teams:
        raise ValueError('Choose teams available in this dataset.')
    if model_name not in bundle['models']:
        raise ValueError('Choose an available model.')
    date=pd.Timestamp(when).normalize()
    end=pd.Timestamp(bundle['metadata']['training_end'])
    if date<=end:
        raise ValueError(f'Choose a date after {end.date()}. These forecast models already learned from matches through that date.')
    if date>end+pd.Timedelta(days=365):
        raise ValueError('Choose a date within one year of the latest data, or update the match history first.')
    history=team_history(clean)
    pair=history[history.Team.isin([home,away])]
    values=fixture_features(pair,home,away,date)
    if values['HomeHistoryCount']==0 or values['AwayHistoryCount']==0:
        raise ValueError('At least one team has no EPL match history in the previous year. Select teams with recent history.')
    x=pd.DataFrame([values],columns=bundle['metadata']['features'])
    model=bundle['models'][model_name]
    p=model.predict_proba(x)[0]
    np.testing.assert_array_equal(model.classes_,[0,1,2])
    if not np.isfinite(p).all() or not np.isclose(p.sum(),1):
        raise ValueError('The model could not produce valid probabilities. Please reload the app.')
    return pd.DataFrame({'Outcome':['Away win','Draw','Home win'],'Probability':p}),values
