#!/usr/bin/env python
# coding: utf-8

# In[1]:


# general
import pandas as pd
import numpy as np
import requests

# plotting
import plotly.express as px
import plotly.graph_objects as go

# stats
from statsmodels.stats.proportion import proportions_ztest


# In[2]:


import addfips
from urllib.request import urlopen
import json
import math
import types
import pkg_resources


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
                                 y = county_hydrogen_fuel['total hydrogen stations'])],
                    layout=go.Layout(title=go.layout.Title(text="Hydrogen Refueling Stations by county")))
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


# In[ ]:





# ## Twitter Analysis

# In[30]:


url = "https://raw.githubusercontent.com/lbarrett24/CA-Election-Data-Challenge/Connor2-branch/STA141B_Project_Sentiment_Analysis_Data.csv"
res = requests.get(url, allow_redirects=True)
with open('cleaned_data.csv','wb') as file:
    file.write(res.content)
sa_df = pd.read_csv('cleaned_data.csv')


# In[31]:


counts_df = sa_df.groupby('Method')['Sentiment'].value_counts().to_frame() # get counts of each sentiment and turn into dataframe
counts_df.rename(columns = {'Sentiment': 'Count'}, inplace=True) # rename columns
counts_df.reset_index(inplace=True) # reset index
total_counts = counts_df.groupby('Method')['Count'].sum().to_list() # count total number of sentiment
total_counts = np.repeat(total_counts, 3) # repeat total counts to get correct dimension
counts_df['Percentage'] = counts_df['Count'] / total_counts # make new column containing percent of each sentiment


# In[32]:


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
fig_all.show()


# In[33]:


# function to format floats into percent format
def format_percent(val: float) -> str:
    '''
    Given a float value between 0 and 1, returns the value as a string in percent format with 2 decimal places.
    Ex: 0.2356 -> '23.56%'
    '''
    trunc_val = round(val, 4) * 100
    format_val = f'{trunc_val:.4}%'
    return format_val


# In[34]:


counts_no_neutral_df = counts_df[counts_df['Sentiment'] != 'Neutral'] # remove rows with neutral sentiment
counts_no_neutral_df = counts_no_neutral_df.drop('Percentage', axis=1) # drop the percentage column. will be remade
total_counts_no_neutral = counts_no_neutral_df.groupby('Method')['Count'].sum().to_list() # get total count of sentiments for each model 
total_counts_no_neutral = np.repeat(total_counts_no_neutral, 2) # repeat total counts to get correct dimension
counts_no_neutral_df['Percentage'] = counts_no_neutral_df['Count'] / total_counts_no_neutral # make new column containing percent of each sentiment
counts_no_neutral_df['Str Percentage'] = counts_no_neutral_df['Percentage'].apply(format_percent) # convert floats into percent-formatted strings


# In[35]:


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
fig_no_neutral.show()


# In[36]:


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
fig_perc_no_neutral.show()


# In[37]:


# assign variables for actual poll results and sample size
poll_negative_count = 6_161_978
poll_positive_count = 4_524_334
poll_sample = poll_negative_count + poll_positive_count


# In[38]:


# create dataframe containig poll data
counts_poll_df = pd.DataFrame({'Method': ['Poll', 'Poll'], 'Sentiment': ['Negative', 'Positive'], 'Count': [poll_negative_count, poll_positive_count]})
counts_poll_df['Percentage'] = counts_poll_df['Count'] / poll_sample # percentage of each sentiment
counts_poll_df['Str Percentage'] = counts_poll_df['Percentage'].apply(format_percent) # string representation of percentages


# In[39]:


# combina sentiment analysis data and poll data
counts_combined_df = pd.concat([counts_no_neutral_df, counts_poll_df])


# In[40]:


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
fig_combined.show()


# In[41]:


alpha = 0.05


# In[42]:


# get positive and negative counts for each model, as well as sample size
r_negative_count, r_positive_count = counts_no_neutral_df[(counts_no_neutral_df['Method'] == 'RoBERTa')]['Count'].to_list()
v_positive_count, v_negative_count = counts_no_neutral_df[(counts_no_neutral_df['Method'] == 'VADER')]['Count'].to_list()
r_sample = r_negative_count + r_positive_count
v_sample = v_negative_count + v_positive_count


# In[43]:


# comparing Negative and No
z_stat_r_neg, p_value_r_neg = proportions_ztest(count = [r_negative_count, poll_negative_count], nobs = [r_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_r_neg


# In[44]:


# comparing Positive and Yes
z_stat_r_pos, p_value_r_pos = proportions_ztest(count = [r_positive_count, poll_positive_count], nobs = [r_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_r_pos


# In[45]:


# comparing Negative and No
z_stat_v_neg, p_value_v_neg = proportions_ztest(count = [v_negative_count, poll_negative_count], nobs = [v_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_v_neg


# In[46]:


# comparing Positive and Yes
z_stat_v_pos, p_value_v_pos = proportions_ztest(count = [v_positive_count, poll_positive_count], nobs = [v_sample, poll_sample], value = 0.0, alternative= 'two-sided')
p_value_v_pos


# In[47]:


methods = ['RoBERTa', 'RoBERTa','VADER', 'VADER'] # list of methods
positions = ['Positive', 'Negative', 'Positive', 'Negative'] # list of positions
pvalues = [round(p_value_r_pos, 3), round(p_value_r_neg, 3), float(f'{p_value_v_pos:.3e}'), float(f'{p_value_v_neg:.3e}')] # list of pvalues
conclusions = ['Reject' if p <= alpha else 'Fail to Reject' for p in pvalues] # list of conclusions


# In[48]:


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
pvalue_table.show()


# In[49]:


import dash as d
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash


# In[50]:


app = d.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.title = 'California 2022 Election Prop 30' 


# In[51]:


app.layout = html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.H1(children="California 2022 Election Prop 30"),
            html.H2(children="STA 141B Final Project - Group 27")
        ]),
        html.Div(children=[
            html.Img(src=d.get_asset_url("/assets/vote.png")),
        ]),
    ]),
    dcc.Tabs(id='tab1', children=[
        dcc.Tab(label="Twitter Sentiment Analysis", children=[
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Graph(id="count_prop30",
                              figure = fig_all)
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dcc.Graph(id="count_no_neutral",
                              figure = fig_no_neutral)
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id="percent_with_prop",
                              figure = fig_combined)
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id="pval_table",
                              figure = pvalue_table)
                ]),
            ]),
        ]),
        dcc.Tab(label="EV Sales", children=[
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
                html.H1(children='EV Purchased by Year'),

                html.Div(children='''
                    Interactive Bar Graph for EVs Purchased
                '''),

                dcc.Graph(
                    id='graph2',
                    figure=fig2
                ),

                html.P(
                    'Although the earliest modern EV was sold in the late 1990s, EVs purchases started to rise in 2011. Areas with higher population density and higher median income like Los Angeles buys more EVs than other counties. The sharp increase in EV purchases in 2021 and 2022 signals the growing popularity of EVs over gas cars.'
                ),
            ]),

                html.Div([
                html.H1(children='Percentage of EVs Purchased'),

                html.Div(children='''
                    Total Breakdown of EVs Purchased by County
                '''),

                dcc.Graph(
                    id='graph3',
                    figure=fig3
                ),  
            ]),

                html.Div([
                html.H1(children='All EV Chargers'),

                html.Div(children='''
                    Chargers in California
                '''),

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
                html.H1(children='EVs Heat Map'),

                html.Div(children='''
                    Click the play button at the bottom to see EVs over time
                '''),

                dcc.Graph(
                    id='graph6',
                    figure=ev_year_fig
                ),

                html.P(
                    'Starting in 2010, there is a massive increase in EVs purchased. Overtime, all counties increased their EV purchases, especially counties with bigger populations.'
                )
            ]),                
        ]),
    ]),
])


# In[ ]:


if __name__ == '__main__':
    app.run_server()

