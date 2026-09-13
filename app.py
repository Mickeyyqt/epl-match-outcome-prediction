"""EPL Match Explorer — Step 5 UI required by the project guide."""
from datetime import date,timedelta
import hashlib
import gzip
import json
import pickle
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scripts.app_support import predict_fixture

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='EPL Match Explorer',page_icon='⚽',layout='wide')
st.markdown('''<style>
.block-container {padding-top:5rem;max-width:1400px;}
h1 {letter-spacing:-1.5px;} h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:white;border:1px solid #dce7e1;border-radius:12px;padding:16px;}
[data-testid="stSidebar"] {border-right:1px solid #d5e1d9;}
div[data-testid="stTabs"] button {font-size:1rem;}
</style>''',unsafe_allow_html=True)


@st.cache_resource
def load_models():
    with gzip.open(ROOT/'app_models.pkl.gz','rb') as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(ROOT/'Cleaned_Dataset.csv',parse_dates=['MatchDate'])


@st.cache_data
def table(name,step=2):
    return pd.read_csv(ROOT/f'reports/step{step}'/name)


def chart(fig):
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',margin=dict(t=45,b=30,l=20,r=20),font=dict(color='#182F29'))
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False})


try:
    data=load_data()
    bundle=load_models()
    if hashlib.sha256((ROOT/'Cleaned_Dataset.csv').read_bytes()).hexdigest()!=bundle['metadata']['canonical_data_sha256']:
        raise ValueError('Data/model version mismatch')
    metrics=table('model_comparison.csv',4)
except (OSError,ValueError,KeyError,EOFError,pickle.UnpicklingError):
    st.error('The app data or models are unavailable. Restore the supplied app files and restart the app.')
    st.stop()

latest=data.MatchDate.max().date()
teams=sorted(set(data.HomeTeam)|set(data.AwayTeam))
current=sorted(set(data[data.Season==data.Season.max()].HomeTeam)|set(data[data.Season==data.Season.max()].AwayTeam))
names=list(bundle['models'])
selected=bundle['metadata']['selected_model_by_validation']
with st.sidebar:
    st.markdown('### EPL Match Explorer')
    st.caption('DATA MINING FINAL PROJECT')
    st.markdown('**Min Thant Htoo**  \nYKPT - 22367')
    st.caption('University of Computer Science, Yangon')
    st.divider()
    model_name=st.selectbox('Prediction model',names,index=names.index(selected),key='model')
    st.caption('Tuned Random Forest was selected using validation macro-F1.')
    with st.expander('Model settings'):
        params=bundle['models'][model_name].named_steps['model'].get_params()
        st.json({k:params[k] for k in ['strategy','C','class_weight','n_estimators','max_depth','min_samples_leaf'] if k in params})
    st.divider()
    st.metric('Matches in dataset',f'{len(data):,}')
    st.caption(f'{data.Season.nunique()} seasons · {len(teams)} clubs · 42 prediction features')
    st.caption(f'Latest match: {latest:%d %b %Y}')
    st.caption('Source: Football-Data. This is a course project; predictions are estimates.')

st.caption('ENGLISH PREMIER LEAGUE · HISTORICAL ANALYSIS & MATCH PREDICTION')
st.title('Explore the past. Estimate the next result.')
st.write('Compare team form, estimate match outcomes and inspect how the models performed.')
explore,predict,analytics,comparison=st.tabs(['Data Exploration','Predictions','Analytics Dashboard','Model Comparison'])

with explore:
    st.subheader('A closer look at a club')
    a,b=st.columns([2,1])
    club=a.selectbox('Find a team',teams,index=teams.index('Arsenal'),key='club')
    seasons=['All seasons']+sorted(data.Season.unique(),reverse=True)
    season=b.selectbox('Season',seasons,key='season')
    matches=data[(data.HomeTeam==club)|(data.AwayTeam==club)].copy()
    if season!='All seasons':
        matches=matches[matches.Season==season]
    if matches.empty:
        st.info('No matches for this club in the selected season. Choose another season or All seasons.')
    else:
        at_home=matches.HomeTeam.eq(club)
        goals_for=matches.FullTimeHomeGoals.where(at_home,matches.FullTimeAwayGoals)
        goals_against=matches.FullTimeAwayGoals.where(at_home,matches.FullTimeHomeGoals)
        points=(goals_for>goals_against)*3+(goals_for==goals_against)
        c=st.columns(4)
        c[0].metric('Matches',len(matches));c[1].metric('Win rate',f'{(goals_for>goals_against).mean():.1%}')
        c[2].metric('Points / match',f'{points.mean():.2f}');c[3].metric('Goals / match',f'{goals_for.mean():.2f}')
        st.caption('Points are derived from match results, before administrative deductions.')
        trend=matches.assign(GoalsFor=goals_for,GoalsAgainst=goals_against).groupby('Season')[['GoalsFor','GoalsAgainst']].mean().reset_index()
        chart(px.line(trend,x='Season',y=['GoalsFor','GoalsAgainst'],markers=True,title=f'{club}: goals per match by season',labels={'value':'Goals per match','variable':'Measure'}))
        st.markdown('#### Match history')
        st.dataframe(matches[['MatchDate','Season','HomeTeam','AwayTeam','FullTimeHomeGoals','FullTimeAwayGoals','FullTimeResult']].sort_values('MatchDate',ascending=False),hide_index=True,width='stretch')

with predict:
    st.subheader('Estimate a match result')
    st.caption(f'Forecast models use completed history through {latest:%d %b %Y}. Select a later date. Team selection describes a matchup; it does not confirm a scheduled fixture.')
    with st.form('fixture'):
        a,b,c=st.columns(3)
        home=a.selectbox('Home team',teams,index=teams.index('Arsenal'),key='home')
        away=b.selectbox('Away team',teams,index=teams.index('Chelsea'),key='away')
        when=c.date_input('Match date',value=max(date.today(),latest+timedelta(days=1)),min_value=latest+timedelta(days=1),max_value=latest+timedelta(days=365),key='when')
        submitted=st.form_submit_button('Predict match',type='primary')
    if submitted:
        try:
            probabilities,features=predict_fixture(data,bundle,model_name,home,away,when)
            best=probabilities.loc[probabilities.Probability.idxmax()]
            st.markdown(f'### {home} vs {away}')
            st.write(f'Most likely outcome: **{best.Outcome}** · {best.Probability:.1%} estimated probability')
            columns=st.columns(3)
            for col,row in zip(columns,probabilities.itertuples()):
                col.metric(row.Outcome,f'{row.Probability:.1%}')
            chart(px.bar(probabilities,x='Outcome',y='Probability',color='Outcome',range_y=[0,1],color_discrete_sequence=['#7b65a8','#d79c3e','#167A64'],title=f'Outcome probabilities · {model_name}'))
            if min(features['HomeHistoryCount'],features['AwayHistoryCount'])<5:
                st.warning('One team has fewer than five recent matches. The prediction uses limited history.')
            if (when-latest).days>14:
                st.warning('The forecast date is more than two weeks beyond the saved history. New results are not included.')
            if home not in current or away not in current:
                st.info('This is a hypothetical matchup: one selected club is absent from the latest season in the dataset.')
            st.dataframe(pd.DataFrame({'Team':[home,away],'Recent matches':[features['HomeHistoryCount'],features['AwayHistoryCount']],
                                        'Recent points / match':[features['HomePointsLast5'],features['AwayPointsLast5']],
                                        'Recent goals / match':[features['HomeGoalsForLast5'],features['AwayGoalsForLast5']]}),hide_index=True,width='stretch')
            st.caption('History includes only earlier dates. A likely outcome is not a guaranteed result. Draw prediction was a weakness in historical testing.')
        except ValueError as exc:
            st.warning(str(exc))

with analytics:
    st.subheader('Patterns in historical matches')
    st.caption('These analyses use the training period, 2001/02–2020/21: 7,600 matches. They describe past patterns, not current league standings.')
    left,right=st.columns(2)
    with left:
        outcomes=table('outcome_distribution.csv').replace({'Outcome':{'H':'Home win','D':'Draw','A':'Away win'}})
        chart(px.bar(outcomes,x='Outcome',y='Count',color='Outcome',title='1 · Match outcome distribution'))
        st.caption('Home wins were the most common outcome. Frequency is not prediction accuracy.')
        goals=table('goal_distribution.csv')
        chart(px.bar(goals,x='TotalGoals',y='Matches',title='3 · Total goals distribution'))
        st.caption('Most matches have modest goal totals, with a smaller tail of high-scoring results.')
        profiles=table('cluster_profiles.csv')
        chart(px.scatter(profiles,x='GoalsForPerMatch',y='GoalsAgainstPerMatch',size='TeamSeasons',color='ProfileName',hover_data=['PointsPerMatch','TeamSeasons'],title='5 · Team-season cluster profiles'))
        st.caption('K-Means grouped 400 completed team-seasons. Profiles summarize past seasons and are not prediction inputs.')
    with right:
        trends=table('season_outcome_percentages.csv').rename(columns={'H':'Home win','D':'Draw','A':'Away win'})
        chart(px.line(trends,x='Season',y=['Home win','Draw','Away win'],title='2 · Outcome shares by season',labels={'value':'Share (%)','variable':'Outcome'}))
        st.caption('Outcome shares change over time; the chart alone does not identify the cause.')
        summary=table('team_summary.csv').query('Matches >= 190').nlargest(10,'PointsPerMatch').sort_values('PointsPerMatch')
        chart(px.bar(summary,x='PointsPerMatch',y='Team',orientation='h',title='4 · Historical points per match'))
        st.caption('Only teams with at least five EPL seasons appear here. All teams remain in the dataset.')
        correlation=table('prematch_correlations.csv').set_index('Unnamed: 0')
        chart(px.imshow(correlation,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='6 · Recent-form feature correlations',aspect='auto'))
        st.caption('Correlation measures association, not causation. Related features may carry overlapping information.')
    with st.expander('Association rules: support, confidence and lift'):
        st.dataframe(table('association_rules.csv'),hide_index=True,width='stretch')
        st.caption('Rules were mined from training matches with five prior matches for both teams. They are historical associations, not validated prediction guarantees.')

with comparison:
    st.subheader('How did the models perform?')
    st.info('Historical test: 1,900 matches from 2021/22–2025/26, using frozen models trained through May 2021. Forecast models in the Predictions tab were later refitted through September 2026 using the same settings. These test scores belong to the earlier frozen versions.')
    st.dataframe(metrics[['Model','Accuracy','MacroPrecision','MacroRecall','MacroF1','MacroOVRAUC']].round(4),hide_index=True,width='stretch')
    chart(px.bar(metrics.melt(id_vars='Model',value_vars=['Accuracy','MacroF1'],var_name='Metric',value_name='Score'),x='Model',y='Score',color='Metric',barmode='group',range_y=[0,1],title='Final test accuracy and macro-F1'))
    st.write('Tuned Random Forest was selected by validation macro-F1. On the final test it achieved **47.05% accuracy** and **0.4321 macro-F1**. Basic Logistic Regression had higher accuracy (**51.68%**) but identified only **1 of 454 draws**, compared with **91** for the selected forest.')
    a,b=st.columns(2)
    with a:
        st.markdown('#### Confusion matrix')
        matrices=json.loads((ROOT/'reports/step4/confusion_matrices.json').read_text())
        chart(px.imshow(matrices['models'][model_name],x=matrices['labels'],y=matrices['labels'],text_auto=True,color_continuous_scale='Blues',labels={'x':'Predicted','y':'Actual'},title=model_name))
    with b:
        st.markdown('#### ROC curves')
        curves=json.loads((ROOT/'reports/step4/roc_curves.json').read_text())[model_name]
        fig=go.Figure()
        for label,v in curves.items():
            fig.add_trace(go.Scatter(x=v['fpr'],y=v['tpr'],mode='lines',name=f'{label} · AUC {v["auc"]:.3f}'))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line={'dash':'dash','color':'gray'},name='Reference'))
        fig.update_layout(xaxis_title='False positive rate',yaxis_title='True positive rate')
        chart(fig)
    with st.expander('Five-fold cross-validation results'):
        st.dataframe(table('cross_validation_summary.csv',4).round(4),hide_index=True,width='stretch')
        st.caption('Ordinary five-fold K-fold within training, with seed 42. Dates are mixed; these tuning scores can be optimistic for future matches. Standard deviations are not confidence intervals.')

st.divider()
st.caption('Min Thant Htoo · YKPT - 22367 · Data source: Football-Data · AI-assisted course project; see project disclosure.')
