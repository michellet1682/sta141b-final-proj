#!/usr/bin/env python
# coding: utf-8

# In[1]:


# general
import pandas as pd
import numpy as np

import addfips
from urllib.request import urlopen
import json
import math
import types
import pkg_resources

import datetime

# plotting
import plotly.express as px
import plotly.graph_objects as go

# stats
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import statsmodels.api as sm 

# dashboard
import dash as d
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash


# In[2]:


app = d.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.title = 'California 2022 Election Prop 30' 


# In[3]:


excel_data = pd.read_excel('Hydrogen_Refueling_Stations_Last updated_10-18-2022.xlsx')
df = pd.DataFrame(excel_data)


# In[4]:


county= df['County'].unique()
#df.groupby('County').unique()
county=np.delete(county, -1)


# In[5]:


county_hydrogen_fuel= pd.DataFrame()

for i in range(len(county)):

    placeholder = pd.DataFrame() #placeholder dataframe
    placeholder['county'] = [county[i]] #state name into state column

    county_rows = df[df['County'] == str(county[i])] #group rows from the same state
    county_sum = county_rows['Fueling Positions'].sum() #sum of each age bin within the same state
    fuel = county_sum.sum() #add the sum of age bins together to get state population

    placeholder['total hydrogen stations'] = [fuel] #state population into state pop column

    county_hydrogen_fuel = pd.concat([county_hydrogen_fuel, placeholder], ignore_index = True) 
    #append placeholder to the state_prop data frame


# In[6]:


def hydrogen_refuel():
    fig1 = go.Figure(data=[go.Bar(x=county_hydrogen_fuel['county'], 
                                 y = county_hydrogen_fuel['total hydrogen stations'])])
    return fig1


# In[7]:


ev_sales_df = pd.read_excel('New_ZEV_Sales_Last_updated_10-18-2022.xlsx')
out_of_state = ev_sales_df[(ev_sales_df['County'] == 'Out Of State') ].index
ev_sales_df.drop(out_of_state , inplace=True)
#ev_sales_df.head(n=6)


# In[8]:


year_df = ev_sales_df.groupby(['Data Year','County']).sum(numeric_only = True)
year_df.reset_index(inplace=True)
year_df.rename(columns = {'Data Year':'Year','Number of Vehicles':'Total Purchased EV'},inplace=True)
#year_df.head()


# In[9]:


fig2 = px.bar(year_df,
             x=year_df['Year'], 
             y='Total Purchased EV', 
             #text_auto=True,
             #log_x = True,
             color = 'County')
fig2.update_xaxes(
        tickangle = 90,
        title_text = "Year",
        title_font = {"size": 20},
        title_standoff = 25,
        tickmode = 'linear')
fig2.update_yaxes(
        tickangle = 90,
        title_text = "Total Purchased EV",
        title_font = {"size": 20},
        title_standoff = 25)
fig2.update_layout(title_text='Total Electric Vehicles Purchased by Year', title_x=0.5)
fig2['layout']['title']['font'] = dict(size=25)
fig2.update_layout()
#fig2.show()


# In[10]:


fig3 = px.pie(year_df, 
              values='Total Purchased EV',
              names='County')
fig3.update_layout(title_text='Percentage of Electric Vehicles Purchased by County', title_x=0.12)
fig3.update_traces(textposition='inside')
fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig3['layout']['title']['font'] = dict(size=25)
#fig3.show()


# In[11]:


ev_chargers = pd.read_csv("EV Chargers_Last updated 01-31-2022.csv")
ev_chargers2 = ev_chargers.drop('Total', axis=1)
ev_chargers2["Public Chargers"] = (ev_chargers2["Public Level 1"] + ev_chargers2["Public Level 2"] + 
                                   ev_chargers2["Public DC Fast"])
ev_chargers2["Private Chargers"] = (ev_chargers2["Shared Private Level 1"] + ev_chargers2["Shared Private Level 2"] + 
                                   ev_chargers2["Shared Private DC Fast"])
ev_chargers2 = ev_chargers2.drop([59])


# In[12]:


charger_list = list(ev_chargers2) 
charger_list.remove('County')
charger_list.remove('Public Chargers')
charger_list.remove('Private Chargers')

ev_df = pd.DataFrame()
for i in range(len(charger_list)):
    placeholder = pd.melt(ev_chargers2, id_vars = ['County'], value_vars = [charger_list[i]])
    ev_df = pd.concat([ev_df, placeholder], ignore_index = True)


# In[13]:


ev_pub = ev_chargers2[["County","Public Level 1", "Public Level 2", "Public DC Fast", "Public Chargers"]]
ev_priv = ev_chargers2[["County","Shared Private Level 1", "Shared Private Level 2", "Shared Private DC Fast", 
                       "Private Chargers"]]


# In[14]:


all_fig = px.bar(ev_df, x="County",y=["value"], color="variable", 
       labels={'_value': 'Count'}, title='All EV Chargers per County')
all_fig.update_layout(legend = dict(title = 'Type of Charger'))


# In[15]:


charger_amount_df = pd.read_csv('EV Chargers_Last updated 01-31-2022.csv')
charger_amount_df = charger_amount_df.dropna()


# In[16]:


af = addfips.AddFIPS()

def create_fips_col(county):
    fips = af.get_county_fips(county, state = 'California')
    return fips

# add 'fips' column to each df
charger_amount_df['fips'] = charger_amount_df['County'].apply(create_fips_col)
charger_amount_df['log_Total'] = charger_amount_df['Total'].apply(math.log)


# In[17]:


with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

target_states = ['06']
counties['features'] = [f for f in counties['features'] if f['properties']['STATE'] in target_states] 


# In[18]:


charger_fig = px.choropleth(
    charger_amount_df,
    geojson = counties,
    locations = 'fips',
    color = 'log_Total',
    scope = 'usa',
    color_continuous_scale= 'agsunset_r',
    hover_name = charger_amount_df['County'],
    hover_data= {'Total': True,
                 'fips': False,
                 'log_Total': False
                 },
    basemap_visible=True
           
)

charger_fig.update_geos(fitbounds = 'locations')
charger_fig.update_layout(height=500,margin={"r":0,"t":0,"l":0,"b":0})
charger_fig.layout.coloraxis.colorbar = {'title': 'Number of EV Chargers',
                                 'x': 0.9,
                                 'tickvals': [0,1,2,3,4,5,6,7,8,9,10, 11],
                                 'ticktext': [0, 2, 5] + [str(int(round(math.exp(val), -1))) for val in range(2,11)]}
#charger_fig.show()


# In[19]:


ev_df = pd.read_csv('New_ZEV_Sales.csv')


# In[20]:


out_of_state = ev_df[ (ev_df['County'] == 'Out Of State') ].index
ev_df.drop(out_of_state , inplace=True)


# In[21]:


ev_year_df = ev_df.groupby(['Data Year','County']).sum(numeric_only = True)
ev_year_df.reset_index(inplace=True)
ev_year_df.rename(columns = {'Number of Vehicles': 'Total', 'Data Year': 'Year'}, inplace = True)
ev_year_df['fips'] = ev_year_df['County'].apply(create_fips_col)


# In[22]:


first_year = ev_year_df['Year'].iloc[0]
last_year = ev_year_df['Year'].iloc[-1]


# In[23]:


counties_lst = ev_year_df['County'].unique()


# In[24]:


for year in range(first_year, last_year + 1):
    for county in counties_lst:
        if not ((ev_year_df['Year'] == year) & (ev_year_df['County'] ==  county)).any():
            temp_df = {'Year': [year], 
                       'County': [county], 
                       'Total': [0], 
                       'fips': [af.get_county_fips(county, state = 'California')]}
            temp_df = pd.DataFrame(temp_df)
            ev_year_df = pd.concat([ev_year_df, temp_df], ignore_index = True)

ev_year_df = ev_year_df.sort_values(by = 'Year')


# In[25]:


def cumulative_sum(row):
    cur_year = row['Year']
    county = row['County']
    cum_sum = ev_year_df[(ev_year_df['Year'] <= cur_year) & (ev_year_df['County'] == county)]['Total'].sum(numeric_only=True)
    return cum_sum


# In[26]:


ev_year_df['Cumulative Total'] = ev_year_df.apply(cumulative_sum, axis=1)


# In[27]:


def log_0(num):
    if num == 0:
        return 0
    else:
        return math.log(num)


# In[28]:


ev_year_df['log_Cum_Total'] = ev_year_df['Cumulative Total'].apply(log_0)


# In[29]:


ev_year_fig = px.choropleth(
    ev_year_df,
    geojson = counties,
    locations = 'fips',
    color = 'log_Cum_Total',
    scope = 'usa',
    color_continuous_scale= 'agsunset_r',
    hover_name = ev_year_df['County'],
    hover_data= {'Cumulative Total': True,
                 'fips': False,
                 'log_Cum_Total': False
                 },
    basemap_visible=True,
    animation_frame='Year'
           
)

ev_year_fig.update_geos(fitbounds = 'locations')
ev_year_fig.update_layout(height=500,margin={"r":0,"t":5,"l":0,"b":0})
ev_year_fig.layout.coloraxis.colorbar = {'title': 'Number of EVs',
                                 'x': 0.9,
                                 'tickvals': [0, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                 'ticktext': [0, 2, 3] + [str(int(round(math.exp(val), -1))) for val in range(2,14)]
                                 }
#ev_year_fig.show()


# ## Luke's OLS

# In[30]:


df_ols = pd.read_csv("alt_fuel_stations_Oct_21_2022.csv")


# In[31]:


#convert object to datetime object
df_ols['Open Date'] = pd.to_datetime(df_ols['Open Date'], format="%Y-%m-%d")


# In[32]:


#extract only the year
charger_year = pd.DatetimeIndex(df_ols['Open Date']).year


# In[33]:


charger_year = pd.DataFrame(charger_year)
charger_year.dropna()
charger_year.astype('Int64')


# In[34]:


charger_year_count = charger_year['Open Date'].value_counts()
charger_year_count = pd.DataFrame(charger_year_count)
charger_year_count.index = charger_year_count.index.astype('Int64')
charger_year_count = charger_year_count.rename(columns={'Open Date':'Total Chargers'})
charger_year_count = charger_year_count.sort_index()


# In[35]:


year_df.drop('County', axis=1, inplace=True)
year_df = year_df.groupby('Year')['Total Purchased EV'].sum()
year_df = pd.DataFrame(year_df)
year_df = year_df.sort_index()


# In[36]:


merged_df = pd.merge(year_df, charger_year_count, left_index=True, right_index=True)


# In[37]:


dta = merged_df[['Total Purchased EV','Total Chargers']]
dta = sm.add_constant(dta) #constant to fit intercept of linear model


# In[38]:


ols1 = sm.OLS(dta['Total Purchased EV'], dta.drop(columns = 'Total Purchased EV')) #Initialize the OLS 
ols1_res = ols1.fit() # ols fitting
summary1 = ols1_res.summary()


# In[39]:


pearsonr(dta['Total Purchased EV'], dta['Total Chargers'])


# ## Andrew stuff

# In[40]:


df_ev = pd.read_csv("New_ZEV_Sales.csv")
# Electric Car sales


# In[41]:


sales_2022 = df_ev[df_ev["Data Year"] == 2022]
# only including car sales from 2022


# In[42]:


counties = sales_2022["County"].unique()
# returns all county names
counties = list(counties)


# In[43]:


county_sales = []
for i in counties:
    county_sales.append(sales_2022.loc[sales_2022['County'] == i, 'Number of Vehicles'].sum())
# Sums all ZEV sales by county


# In[44]:


sales_by_county = pd.DataFrame(county_sales, counties)
# Data Frame of ZEV sales in 2022 by county


# In[45]:


df1 = pd.read_csv('CA_county_pop.csv')
# Load in CA 2022 Population data by County


# In[46]:


county_data = sales_by_county
county_data.rename(columns = {0:'2022 Sales'}, inplace = True)
# Renames Sales Column


# In[47]:


county_data["2022 Sales"] = county_data["2022 Sales"]/list(df1.iloc[:, 0])
# Takes total sales of ZEV vehicles in 2022 by county divided by total population of county
county_data.rename(columns = {'2022 Sales':'sales'}, inplace = True)


# In[48]:


df2 = pd.read_csv('Prop 30 Election Results - Sheet1.csv')
# Loads Election Data by County.  This was taken from NYT article table which was converted to excel which was converted ot csv
# Link of nyt article: https://www.nytimes.com/interactive/2022/11/08/us/elections/results-california-proposition-30-electric-vehicle-incentive-tax.html


# In[49]:


x = list(df2.iloc[:, 1])
x[18] = 58 #original file had 58% instead of 58


# In[50]:


x = [int(i)/100 for i in x]
# changes voter data into integers


# In[51]:


county_data['vote'] = x
# adds No vote data to County Data Frame


# In[52]:


q1y=county_data['vote'].quantile(0.25)
q3y=county_data['vote'].quantile(0.75)
IQRy=q3y-q1y
y_outliers = county_data['vote'][((county_data['vote']<(q1y-1.5*IQRy)) | (county_data['vote']>(q3y+1.5*IQRy)))]
# Finds outliers of voter data


# In[53]:


q1x=county_data['sales'].quantile(0.25)
q3x=county_data['sales'].quantile(0.75)
IQRx=q3x-q1x
x_outliers = county_data['sales'][((county_data['sales']<(q1x-1.5*IQRx)) | (county_data['sales']>(q3x+1.5*IQRx)))]
# finds outliers of ZEV sales data


# In[54]:


x_outliers = pd.DataFrame(x_outliers)
y_outliers = pd.DataFrame(y_outliers)


# In[55]:


outliers = [y_outliers.index[0]] + [x_outliers.index[0] ] + [x_outliers.index[1] ] + [x_outliers.index[2] ] + [x_outliers.index[3] ] + [x_outliers.index[4] ]
#indexes of all outliers


# In[56]:


county_data = county_data.drop(outliers)
county_data = sm.add_constant(county_data)
# adds B0 to data


# In[57]:


ols = sm.OLS(county_data['vote'], county_data.drop(columns = 'vote')) #Initialize the OLS 
ols_res = ols.fit() # ols fitting
# fits data


# In[58]:


summary = ols_res.summary()
#summary of Data
#print(summary)


# In[59]:


ols_res.rsquared
#r squared of data


# In[60]:


county_data= county_data.drop(columns=['const'])
#removes B0's from data


# In[61]:


no_votes = px.scatter(county_data,x ="sales", y ="vote", trendline = 'ols', trendline_scope = 'overall', title = "No Votes by 2022 ZEV Sales")

# Plots Data with trend line


# ## Twitter Analysis

# In[62]:


sa_df = pd.read_csv('STA141B_Project_Sentiment_Analysis_Data.csv')


# In[63]:


counts_df = sa_df.groupby('Method')['Sentiment'].value_counts().to_frame() # get counts of each sentiment and turn into dataframe
counts_df.rename(columns = {'Sentiment': 'Count'}, inplace=True) # rename columns
counts_df.reset_index(inplace=True) # reset index
total_counts = counts_df.groupby('Method')['Count'].sum().to_list() # count total number of sentiment
total_counts = np.repeat(total_counts, 3) # repeat total counts to get correct dimension
counts_df['Percentage'] = counts_df['Count'] / total_counts # make new column containing percent of each sentiment


# In[64]:


# bar plot containing all 3 sentiments for the two models
fig_all = px.bar(data_frame=counts_df, 
                 x = 'Sentiment', 
                 y = 'Count', 
                 color = 'Method', 
                 barmode = 'group', 
                 title = 'Count of Prop. 30 Sentiments on Twitter', 
                 text = 'Count')
fig_all.update_traces(textposition = 'inside', textfont_color = 'white')
fig_all.update_layout(xaxis = {'categoryorder': 'array', 'categoryarray': ['Negative', 'Neutral', 'Positive']},
                      font_size = 14)
#fig_all.show()


# In[65]:


# function to format floats into percent format
def format_percent(val: float) -> str:
    '''
    Given a float value between 0 and 1, returns the value as a string in percent format with 2 decimal places.
    Ex: 0.2356 -> '23.56%'
    '''
    trunc_val = round(val, 4) * 100
    format_val = f'{trunc_val:.4}%'
    return format_val


# In[66]:


counts_no_neutral_df = counts_df[counts_df['Sentiment'] != 'Neutral'] # remove rows with neutral sentiment
counts_no_neutral_df = counts_no_neutral_df.drop('Percentage', axis=1) # drop the percentage column. will be remade
total_counts_no_neutral = counts_no_neutral_df.groupby('Method')['Count'].sum().to_list() # get total count of sentiments for each model 
total_counts_no_neutral = np.repeat(total_counts_no_neutral, 2) # repeat total counts to get correct dimension
counts_no_neutral_df['Percentage'] = counts_no_neutral_df['Count'] / total_counts_no_neutral # make new column containing percent of each sentiment
counts_no_neutral_df['Str Percentage'] = counts_no_neutral_df['Percentage'].apply(format_percent) # convert floats into percent-formatted strings


# In[67]:


# bar plot of positive and negative sentiments for each model
fig_no_neutral = px.bar(data_frame=counts_no_neutral_df, 
                 x = 'Sentiment', 
                 y = 'Count', 
                 color = 'Method', 
                 barmode = 'group', 
                 title = 'Count of Prop. 30 Sentiments on Twitter (excluding Neutral)', 
                 text = 'Count')
fig_no_neutral.update_traces(textposition = 'inside', textfont_color = 'white')
fig_no_neutral.update_layout(xaxis = {'categoryorder': 'array', 'categoryarray': ['Negative', 'Positive']},
                      font_size = 14)
#fig_no_neutral.show()


# In[68]:


# bar plot of percentages of positive and negative sentiments for each model
fig_perc_no_neutral = px.bar(data_frame=counts_no_neutral_df, 
                 x = 'Sentiment', 
                 y = 'Percentage', 
                 color = 'Method', 
                 barmode = 'group', 
                 title = 'Percentage of Prop. 30 Sentiments on Twitter (excluding Neutral)', 
                 text = 'Str Percentage')
fig_perc_no_neutral.update_traces(textposition = 'inside', textfont_color = 'white')
fig_perc_no_neutral.update_layout(xaxis = {'categoryorder': 'array', 'categoryarray': ['Negative', 'Positive']},
                      font_size = 14)
#fig_perc_no_neutral.show()


# In[69]:


# assign variables for actual poll results and sample size
poll_negative_count = 6_161_978
poll_positive_count = 4_524_334
poll_sample = poll_negative_count + poll_positive_count


# In[70]:


# create dataframe containig poll data
counts_poll_df = pd.DataFrame({'Method': ['Poll', 'Poll'], 'Sentiment': ['Negative', 'Positive'], 'Count': [poll_negative_count, poll_positive_count]})
counts_poll_df['Percentage'] = counts_poll_df['Count'] / poll_sample # percentage of each sentiment
counts_poll_df['Str Percentage'] = counts_poll_df['Percentage'].apply(format_percent) # string representation of percentages


# In[71]:


# combina sentiment analysis data and poll data
counts_combined_df = pd.concat([counts_no_neutral_df, counts_poll_df])


# In[72]:


# bar plot of percentages of positive and negative sentiment of each model and poll results
fig_combined = px.bar(data_frame=counts_combined_df, 
                 x = 'Sentiment', 
                 y = 'Percentage', 
                 color = 'Method', 
                 barmode = 'group', 
                 title = 'Percentage of Prop. 30 Sentiments on Twitter (excluding Neutral) compared with Poll results', 
                 text = 'Str Percentage')
fig_combined.update_traces(textposition = 'inside', textfont_color = 'white')
fig_combined.update_layout(xaxis = {'categoryorder': 'array', 'categoryarray': ['Negative', 'Positive']},
                      font_size = 14)
#fig_combined.show()


# In[73]:


alpha = 0.05


# In[74]:


# get positive and negative counts for each model, as well as sample size
r_negative_count, r_positive_count = counts_no_neutral_df[(counts_no_neutral_df['Method'] == 'RoBERTa')]['Count'].to_list()
v_positive_count, v_negative_count = counts_no_neutral_df[(counts_no_neutral_df['Method'] == 'VADER')]['Count'].to_list()
r_sample = r_negative_count + r_positive_count
v_sample = v_negative_count + v_positive_count


# In[75]:


# comparing Negative and No
z_stat_r_neg, p_value_r_neg = proportions_ztest(count = [r_negative_count, poll_negative_count], nobs = [r_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_r_neg


# In[76]:


# comparing Positive and Yes
z_stat_r_pos, p_value_r_pos = proportions_ztest(count = [r_positive_count, poll_positive_count], nobs = [r_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_r_pos


# In[77]:


# comparing Negative and No
z_stat_v_neg, p_value_v_neg = proportions_ztest(count = [v_negative_count, poll_negative_count], nobs = [v_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_v_neg


# In[78]:


# comparing Positive and Yes
z_stat_v_pos, p_value_v_pos = proportions_ztest(count = [v_positive_count, poll_positive_count], nobs = [v_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_v_pos


# In[79]:


methods = ['RoBERTa', 'RoBERTa','VADER', 'VADER'] # list of methods
positions = ['Positive', 'Negative', 'Positive', 'Negative'] # list of positions
pvalues = [round(p_value_r_pos, 3), round(p_value_r_neg, 3), float(f'{p_value_v_pos:.3e}'), float(f'{p_value_v_neg:.3e}')] # list of pvalues
conclusions = ['Reject' if p <= alpha else 'Fail to Reject' for p in pvalues] # list of conclusions


# In[80]:


# make table containing p-value data
pvalue_table = go.Figure(data = [go.Table(
    header = dict(values = ['Method', 'Sentiment', 'P-value', 'Conclusion'],
                  fill_color = 'grey',
                  line_color = 'darkslategray',
                  font = dict(color = 'white', size = 14),
                  align = 'left'),
    cells = dict(values = [methods, positions, pvalues, conclusions],
                 fill_color = [['white', 'lightgrey'] * 2],
                 line_color = 'darkslategray',
                 align = 'left')
)])
#pvalue_table.show()


# In[81]:


import dash as d
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash


# In[82]:


methodology = """
The [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model we used is a deep learning model that was trained on “~124 million tweets from January 2018 to December 2021” and is finetuned for Twitter sentiment analysis. This model uses neural networks and is much more powerful than the VADER model, so we expected it to perform much better. Thus, the sentiments determined by the RoBERTa model are what we used to answer if the opinions on Twitter are representative of election results.

The [VADER](https://www.nltk.org/_modules/nltk/sentiment/vader.html) model is from the NLTK package and is a “lexicon and simple rule-based model for sentiment analysis” .[¹](https://towardsdatascience.com/social-media-sentiment-analysis-in-python-with-vader-no-training-required-4bc6a21e87b8) It uses a dictionary to determine the sentiment scores of individual words and phrases and then calculates a compound score to rate the sentiment of the entire text. Since this model is not trained on real data and only follows relatively basic rules, its performance was expected to be worse than the RoBERTa model.

Before running all of the Tweets through the two models, we did some basic preprocessing on the texts. Any user mention (i.e “@johnsmith123”) was replaced with “@user”. Additionally, any link was replaced with “http”. After completing these basic changes, all of the Tweets were run through the two models.

After determining the sentiments of the Tweets, we removed all the Tweets that had Neutral sentiment and found the proportions of Positive and Negative sentiments of the remaining Tweets for each model. We then compared those proportions with the actual poll results using two-proportion z-tests.

"""


# In[83]:


app.layout = html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.H1(children="California 2022 Election Prop 30"),
                html.H2(children="STA 141B Final Project - Group 27")
            ]),
        ], className="one-half column", id="title"),
    ]),
    dcc.Tabs(id='tab1', children=[
        dcc.Tab(label="Project Context", children=[
            html.Div(children=[
                html.H2("Members"),
                html.P("Lukas Barrett"),
                html.P("Kaleem Ezatullah"),
                html.P("Andrew Muench"),
                html.P("Michelle Tsang"),
                html.P("Connor Young")
            ]),
            html.Div(children=[
                html.Div(children=[
                    html.H1("Proposition 30"),
                    html.P("Prop 30 will increase taxes by 1.5% for those with a personal income of $2 million or more. These taxes will go towards funding electric car rebates, building charging stations, and wildfire prevention."),
                    html.P("Yes - This will help improve air quality in the state, not only by having more firefighting training to stop wildfires, but reduce gas emissions from cars by switching to zero emission electric vehicles. Funding from the taxes will support low income communities with rebates and incentives from buying EVs."),
                    html.P("No - With inflation, this is not a good time to be raising taxes as it will further disrupt the state’s unstable finances. The funding from the taxes will instead help big corporations like Lyft, who are required to increase the number of zero emission vehicles used.")
                ]),
                
                html.Div(children=[
                    html.H1("Questions"),
                    html.H4("We will be focusing on the ZEV/EV sales and voting data of Prop 30. Here are our research questions."),
                    html.P("Is Twitter an accurate representation of election results?"),
                    html.P("What is the current climate of Zero Emissions Vehicles (ZEV) & Electric Vehicles(EV) & their chargers in California? How many more charging stations would be considered beneficial if Prop 30 is passed?"),
                    html.P("How prepared is CA to completely shift from gas to ZEV/EV? By county?"),
                    html.P("Is there a relationship between Electric Vehicle ownership and support for Prop 30?")
                ]),
                
                html.Div(children=[
                    html.H1("Data Sources"),
                    html.A("Population of Each County in California", href="https://worldpopulationreview.com/states/california/counties",
                          target="_blank"),
                    html.Div([
                        html.A("Prop 30 Results for Each County", href="https://www.nytimes.com/interactive/2022/11/08/us/elections/results-california-proposition-30-electric-vehicle-incentive-tax.html",
                               target="_blank"),
                    ]),
                    html.Div([
                        html.A("Datasets of ZEV/EVs and EV Chargers for California", href="https://www.energy.ca.gov/files/zev-and-infrastructure-stats-data",
                              target="_blank")
                    ]),
                    html.Div([
                        html.P([
                            html.Br()
                        ]),
                    ]),
                ]),
                
                html.Div(children=[
                    html.H1("Github Repos"),
                    html.Div([
                        html.A("All code used for this project", href="https://github.com/lbarrett24/CA-Election-Data-Challenge/tree/main",
                              target="_blank")
                    ]),
                    html.Div([
                        html.A("Deployment connected to this Github Repo", href="https://github.com/michellet1682/STA141B-final-project/tree/main",
                              target="_blank")
                    ]),
                ]),
            ]),
        ]),
        
        dcc.Tab(label="Twitter Sentiment Analysis", children=[
            html.Div(children=[
                html.H1(children="Intro"),
                html.Div(children=[
                    html.P("""To determine if Twitter is an accurate representation of election results, we used web scraping and sentiment analysis. We scrapped 2,612 Tweets related to Proposition 30 which were posted between July 1, 2022 and November 8, 2022 using the SNScrape package. We then used two sentiment analysis models (RoBERTa and VADER) to determine whether a Tweet was in support of Prop. 30 or not, as well as comparing the effectiveness of each model. For our analysis, a Positive sentiment corresponds to a Yes on Prop. 30 and a Negative sentiment corresponds to a No.
                """)
                ]),        
            ]),
            
            html.Div(children=[
                dcc.Markdown(children=methodology),
            ]),
            
            html.Div(children=[
                html.H1("Figure 1"),
                dcc.Graph(id="count_prop30",
                          figure = fig_all),
            ]),
            
            html.Div(children=[
                html.H1("Figure 2"),
                dcc.Graph(id="count_no_neutral",
                              figure = fig_no_neutral),
                html.P("Based on the plot in Figure 1, we see that the VADER model resulted in primarily Positive sentiment, followed by Negative, and then Neutral. However, the RoBERTa model resulted in primarily Neutral sentiment, followed by Negative, and then Positive. If we remove the Neutral sentiments as in Figure 2, we see that the RoBERTa model resulted in slightly more Negative than Positive sentiments, whereas the VADER model resulted in over twice as many Positive sentiments than Negative. Overall, we see that the two models resulted in opposite majorities and that the VADER model had a much larger majority."),
                html.P("We will now compare the results of the two models with the actual poll results.")
            ]),
            
            html.Div(children=[
                html.H1("Figure 3"),
                dcc.Graph(id="percent_with_prop",
                              figure = fig_combined),
                html.P("Based on the plot in Figure 3, we see that the proportion of each sentiment from the RoBERTa model are very similar to the proportions of the actual poll results. The proportion of each sentiment from the VADER model, however, are very different from the actual poll results."),
                html.P("To determine the statistical significance of these visible similarities and differences, we will perform several two-proportion z-tests. Our null hypothesis will be that the proportion of a specific sentiment from a model is equal to the proportion of that same sentiment from the ballot results. Our null hypothesis will be that the proportions are different."),
                html.P("The results of these z-tests are shown in the table below.")
            ]),
            
            html.Div(children=[
                html.H1("Figure 4"),
                dcc.Graph(id="pval_table",
                              figure = pvalue_table),
                html.P("From the table in Figure 4, we see that we failed to reject the null hypothesis for both tests involving the RoBERTa model. Thus, there was not significant evidence to conclude that the sentiment proportions from the RoBERTa model were different from the sentiment proportions from the poll."),
                html.P("We also see that we rejected the null hypothesis for both tests involving the VADER model. Thus, there was significant evidence to conclude that the sentiment proportions from the VADER model were different from the sentiment proportions from the poll"),
                html.P("From these results, we can conclude that RoBERTa model is far more accurate than the VADER model, and that the proportion of opinions on Twitter can accurately represent the results of an election.")
            ]),
        ]),
        
        dcc.Tab(label="ZEV/EV Sales and Electric Chargers", children=[
            html.Div([
                html.H1(children='Hydrogen Refueling'),

                html.Div(children='''
                    Other counties not listed are 0.
                '''),

                dcc.Graph(
                    id='graph1',
                    figure=hydrogen_refuel()
                ),

                html.P(
                    'Southern California and the Bay area are the leaders by far when it comes to avaialbility of hydrogen refueling stations. Approximately 50% of all the hydrogen stations in CA are private and not accessible to the public which are used specifically used only for the private sector. These private stations are typically used by industrial purposes in refining and chemical processes. With a Prop30 approval, hydrogen vehicles would likely increase in number requiring more public infrastructure for refueling. This would also allow the industrial sector to tap into these stations as well making working conditions more convenient. Hydrogen fuel cells are major proponents for the ZEV sector since their only waste is water and air.'
                ),
            ]),
            # New Div for all elements in the new 'row' of the page
            html.Div([
                html.H1(children='Total Electric Vehicles Purchased by Year'),

                dcc.Graph(
                    id='graph2',
                    figure=fig2
                ),

                html.P(
                    'Although the earliest modern EV was sold in the late 1990s, EVs purchases started to rise in 2011. Areas with higher population density and higher median income like Los Angeles will on average buy more EVs than other counties. The sharp increase in EV purchases in 2021 and 2022 signals the growing popularity of EVs over gas cars.'
                ),
            ]),

                html.Div([
                html.H1(children='Percentage of Electric Vehicles Purchased by County'),

                dcc.Graph(
                    id='graph3',
                    figure=fig3
                ),  
            ]),

                html.Div([
                html.H1(children='EV Charging Station Types by County'),

                dcc.Graph(
                    id='graph4',
                    figure=all_fig
                ),

                html.P(
                    'Southern California and the Bay Area are EV Charger leaders by far serving the majority of EV owners in the state. A potential benefit of passing Prop30 would be the acquisition of more publicly owned EV chargers. Currently the state leading counties have over ninety-percent of the EV chargers as private entitites which are subject to the private sectors ruling. If Prop30 is withheld however,  it may allow the private sector to benefit more and generate more taxable income that would benefit the state. Private networks may cost more but,  in return they need less maintenance since there is less wear and tire on the infrastructure and allows for faster charging since it is limited to  those individuals who have access to that particular private network. Public networks may allow for more potential users but, it slows the charging speeds down substantially since the chargers would be pulling more power.'
                ),
            ]),

                html.Div([
                html.H1(children='Chargers Heat Map'),

                html.Div(children='''
                    Map of Chargers in California
                '''),

                dcc.Graph(
                    id='graph5',
                    figure=charger_fig
                ),

                html.P(
                    'Locations such as Los Angeles and San Bernardino has much more chargers than other counties. This aligns with the visuals above since the two places has much more EVs than other counties. It appears that a county might have more chargers if more residents there own an EV.'
                )
            ]),   

                html.Div([
                html.H1(children='EV Heat Map Over Time'),

                dcc.Graph(
                    id='graph6',
                    figure=ev_year_fig
                ),

                html.P(
                    'Starting in 2010, there is a massive increase in EVs purchased. Overtime, all counties increased their EV purchases, especially counties with bigger populations.'
                )
            ]),
            
                html.Div([
                    html.H1("Total Purchased EVs vs Total Chargers"),
                    html.Div([
                        html.P(str(summary1), style={'whiteSpace': 'break-spaces'}),
                        html.H3("Analysis"),
                        html.P("With an adjusted R squared value of 0.615, this tells us that about 61.5% of the variability on the number of chargers can be explained by the amount of EVs purchased. Although generally speaking we would want to obtain a model wit more accuracy, considering there may be many more factors that can be affecting the variability (and the huge BIC and AIC values) in this case, and adjusted r-squared of a little over 60% may be sufficient with the data we have in hand."),
                        html.P("There is an error for high multicollinearity which suggests that total purchased EVs and total chargers have a high correlation."),
                        html.P("After conducting a Pearson correlation test, we get r = 0.797 which explains the high multicollinearity. If there was more data with more variables, we could conduct a more insightful analysis on what affects total chargers in California.")
                    ])
            ]),            
        ]),
        dcc.Tab(label="2022 ZEV Sales vs. Prop 30 Votes", children=[
            html.Div([
                html.H1("Datasets"),
                html.P("The data that we analyzed consist of 2 columns, sales and vote. The sales column is the sum of ZEV sales in 2022 divided by the populations of the counties at 2022. The vote column is the percentage of No votes per county"),
                html.P("The goal of the data is to see if there is a connection between the ZEV sales in 2022 and the opposition to Prop 30."),
                html.P("First, we removed the outliers of the data based on the vote percentage and ZEV sales which were San Francisco county ,Marin county, Monterey county, Orange County, San Mateo County, and Santa Clara County. Then, we fitted the vote data to the ZEV sales data.")
            ]),
            
            html.Div([
                html.P(str(summary), style={'whiteSpace': 'break-spaces'}),
                html.P("The model is weak with only 18.3% of the data explained by the fit.  But the data seems to show a downward trend.  This would mean that as ZEV sales increase, then the No vote for Prop 30 decreases in the county.")
            ]),
            
            html.Div([     
                dcc.Graph(id="linreg",
                         figure=no_votes),
                html.P("")
            ]),
            
        ]),
    ]),
])


# In[84]:


if __name__ == '__main__':
    app.run_server()

