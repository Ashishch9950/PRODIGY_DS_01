#!/usr/bin/env python
# coding: utf-8

# # Task-01
# 
# ## Create a bar chart or histogram to visualize the distribution of a categorical or continuous variable, such as the distribution of age or gender in a population.

# ## Importing all the dependencies 

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# ## Initial loading and lookup of the dataset 

# In[3]:


df_total = pd.read_csv(r"C:\Users\Ashish Chauhan\Downloads\Total_Population\Total_Population Data.csv")

df_male = pd.read_csv(r"C:\Users\Ashish Chauhan\Downloads\Male_Population\Male_Population Data.csv")

df_female = pd.read_csv(r"C:\Users\Ashish Chauhan\Downloads\Female_Population\Female_Population Data.csv")


# In[4]:


df_total.head(3)


# In[5]:


df_male.head(3)


# In[6]:


df_female.head(3)


# In[7]:


df_total_copy = df_total.copy()
df_male_copy = df_male.copy()
df_female_copy = df_female.copy()


# In[8]:


df_total_copy.isnull().sum().sum()


# In[9]:


df_total_copy[df_total_copy['2018'].isnull()]


# ## Cleaning and removing duplicates

# In[10]:


# ## Making a copy of the datasets so that the original data stays intact

# df_total_copy = df_total.copy()
# df_male_copy = df_male.copy()
# df_female_copy = df_female.copy()


# df_total_copy.isnull().sum()                                   ## checking for null values
# df_male_copy.isnull().sum()
# df_female_copy.isnull().sum()


# df_total_copy[df_total_copy["1960"].isnull()]                  ## looking for the rows containing nulls
# df_male_copy[df_total_copy["1960"].isnull()]
# df_female_copy[df_total_copy["1960"].isnull()]


# df_total_copy.drop(index=[110,196], inplace=True)              ## dropping the rows with index [110, 196] from the 3 datasets
# df_male_copy.drop(index=[110,196], inplace=True)
# df_female_copy.drop(index=[110,196], inplace=True)


# ## Checking for duplicates in the datasets

df_total_copy.duplicated().sum()
df_male_copy.duplicated().sum()
df_female_copy.duplicated().sum()


# ### Oberservation:
# 
# - Row no 110 has no values for each respective years, so this particular row has no use and should be removed
# - While row no 196 also have most of the values missing, the geographical location of the countries are very different so we cannot replace the missing values either, we will drop this row as well.
# - All the 3 datasets doesn't contain any duplicated values

# In[11]:


df_total_copy.head(3)


# ## Making subsets of the datasets, including the data only for asian countries that are present in the dataset
# 
# ## - Analyzing data for the asian countries

# In[12]:


## List of asian countries 

asian_countries = ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh",
    "Bhutan", "Brunei", "Cambodia", "China", "Cyprus",
    "Georgia", "India", "Indonesia", "Iran", "Iraq",
    "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait",
    "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives",
    "Mongolia", "Myanmar (Burma)", "Nepal", "North Korea", "Oman",
    "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
    "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan",
    "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates",
    "Uzbekistan", "Vietnam", "Yemen"]


## Creating subset of data to only include asian countries
df_total_subset = df_total_copy[df_total_copy["Country Name"].isin(asian_countries)]

df_male_subset = df_male_copy[df_male_copy["Country Name"].isin(asian_countries)]

df_female_subset = df_female_copy[df_female_copy["Country Name"].isin(asian_countries)]


print(f"Shape of the total_population dataset: {df_total_subset.shape}")               ## Confirming their shapes to be alike
print(f"Shape of the male_population dataset: {df_male_subset.shape}")
print(f"Shape of the female_population dataset: {df_female_subset.shape}")



# ## Which are the most populated asian countries by the year 2022. Show the analysis of total_population as well as male and female population

# In[13]:


# Make a copy of the DataFrames
df_total_subset_copy = df_total_subset.copy()
df_male_subset_copy = df_male_subset.copy()
df_female_subset_copy = df_female_subset.copy()

# Sort the copied DataFrames
df_total_subset_copy.sort_values("2022", ascending=False, inplace=True)
df_male_subset_copy.sort_values("2022", ascending=False, inplace=True)
df_female_subset_copy.sort_values("2022", ascending=False, inplace=True)


total_population_countries = df_total_subset_copy["Country Name"].values    ## storing the country names from the total population subset
male_population_countries = df_male_subset_copy["Country Name"].values
female_population_countries = df_female_subset_copy["Country Name"].values


## defining a function to segregate the countries from total_population
def total_population(total_population_countries):
    count = 0
    for i in total_population_countries:
        count += 1
        if count > 5:
            break 
        else:
            print(i)

## defining a function to segregate the countries from male_population
def male_population(male_population_countries):
    count = 0
    for i in male_population_countries:
        count += 1
        if count > 5:
            break 
        else:
            print(i)

## defining a function to segregate the countries from female_population
def female_population(female_population_countries):
    count = 0
    for i in female_population_countries:
        count += 1
        if count > 5:
            break 
        else:
            print(i)


print("Based on Total Population\nTop 5 most populated asian countries by year 2022 are:")
total_population(total_population_countries)
print()
print("Based on Male Population\nTop 5 most populated asian countries by year 2022 are:")
male_population(male_population_countries)
print()
print("Based on Female Population\nTop 5 most populated asian countries by year 2022 are:")
female_population(female_population_countries)

# df_female_subset_copy.head()


# ### Inference:
# 
# ### Based on 2022 population:
# 
# - As we can see from the above analysis, the top 5 most populated countries based on total population data are: India, China, Indonesia, Pakistan, Bangladesh
# - The top 5 most populated asian countries based on male population are: India, China, Indonesia, Pakistan, Bangladesh
# - But the top 5 most populated asian countries based on female population are: China, India, Indonesia, Pakistan, Bangladesh
# - Here we can see that female population is more in China than other asian countries

# ## What is the distribution of gender in the most populated asian countries?

# ### Reframing the dataset to work upon easily:
# - ### Creating only 3 columns : [Country, Year, Population]
# - ### Plotting boxplots for male and female from 1960-2022 to understand the distribution of gender over the years and across the countries

# In[14]:


df_male_subset.dtypes


# In[15]:


years = df_male_subset.select_dtypes([int, float]).columns
years


# In[16]:


## segregating the years from the df_male_subset
years = df_male_subset.select_dtypes([int, float]).columns    

## list of top 5 asian countries
country_names = ["India", "China", "Indonesia", "Pakistan", "Bangladesh"]

## creating a list of population for each consecutive countries
male_country_population = [
    df_male_subset_copy.iloc[0, 4:].tolist(),
    df_male_subset_copy.iloc[1, 4:].tolist(),
    df_male_subset_copy.iloc[2, 4:].tolist(),
    df_male_subset_copy.iloc[3, 4:].tolist(),
    df_male_subset_copy.iloc[4, 4:].tolist()
]

maledata = []

for country, male_population in zip(country_names, male_country_population):
    maledata.extend([{"Country" : country, "Year" : year, "Population" : malepop} for year, malepop in zip(years, male_population)])

male_data = pd.DataFrame(maledata)


female_country_population = [
    df_female_subset_copy.iloc[0, 4:].tolist(),
    df_female_subset_copy.iloc[1, 4:].tolist(),
    df_female_subset_copy.iloc[2, 4:].tolist(),
    df_female_subset_copy.iloc[3, 4:].tolist(),
    df_female_subset_copy.iloc[4, 4:].tolist()
]

femaledata = []

for country, femalepopulation in zip(country_names, female_country_population):
    femaledata.extend([{"Country" : country, "Year" : year, "Population" : femalepop} for year, femalepop in zip(years, femalepopulation)])

female_data = pd.DataFrame(femaledata)

print(male_data.head())
print(female_data.head())


# ### Plotting Boxplots:

# In[49]:


## setting up the y_ticks and y_ticklabel to match the real data

min_population = 24795178.0
max_population = 731180498.0

num_ticks = 15

y_ticks = np.linspace(min_population, max_population, num_ticks).astype(int)    ## using linspace to divide the min and max population equally

y_tick_labels = [f"{val / 1000000:.2f}M" for val in y_ticks]                    ## converting the y_ticks value into Million to be used as y_ticklabel

# --------------------------------------------------------------------------------------------------------------------------------------- #

fig, axes = plt.subplots(1,2, figsize = (15,6))

sns.set_style("whitegrid")
sns.boxplot(x = male_data["Country"], y = male_data["Population"], hue = male_data["Country"], ax = axes[0])
axes[0].set_title("Boxplot for Male Asian Population")
axes[0].set_yticks(y_ticks)
axes[0].set_yticklabels(y_tick_labels)


sns.boxplot(x = female_data["Country"], y = female_data["Population"], hue = female_data["Country"], ax = axes[1])
axes[1].set_title("Boxplot for Female Asian Population")
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels(y_tick_labels)


plt.tight_layout()
plt.show()


# ### Oberservation From Above Charts:
# 
# - Countries like India and China have population spread that surpasses countries like Indonesia, Pakistan, Bangladesh.
# - In male population:
# 1. India has overall uniformly distributed data
# 2. China's male population distribution is slightly shifted towards the higher end
# - In female population:
# 1. India's female population is slightly shifted towards the higher end
# 2. China's female population is evenly distributed

# ### Plotting Histogram for each country based on Male and Female Population:

# In[17]:


## added a new column gender in both the datasets
male_data["Gender"] = "M"
female_data["Gender"] = "F"

## merged the datasets for better analysis
merged_data = pd.concat([male_data, female_data], axis = 0)
merged_data["Gender"].value_counts()
merged_data["Year"] = merged_data["Year"].astype(int)

# ------------------------------------------------------------------------------------------------------------------------------ #

unique_countries = male_data["Country"].unique()

sns.set_style("whitegrid")
fig, axes = plt.subplots(3,2, figsize = (15,10))

## running a loop to plot the histograms
for i, country in enumerate(unique_countries):
    data = merged_data[merged_data["Country"] == country]
    row = i // 2
    col = i % 2
    sns.histplot(x = data["Population"], bins = 8, kde = True, ax = axes[row, col], color="blue", hue = data["Gender"], palette="rocket")
    axes[row, col].set_title(f"Histogram for {country} Population Gender-wise")
    axes[row, col].set_xticks(np.linspace(24795178,731180498,9).astype(int))
    axes[row, col].set_xticklabels(f"{val/1000000:.2f}M"for val in np.linspace(24795178,731180498,9).astype(int))


plt.grid(False)
plt.tight_layout()
plt.show()


# ### Oberservation From Above Charts:
# 
# - Countries like India and China have population spread that surpasses countries like Indonesia, Pakistan, Bangladesh.
# - For India:
# 1. The male population has a slight peak between 200-300 Million with an overall evenly distributed data
# 2. The female population has more peak between 500-650 Million with an overall positive skewed distribution
# - For China:
# 1. The male population is highly right skewed with most of the population lying between 500-700 Million
# 2. The female population has slight peaks around 200-280 Million with overall evenly distributed data
# - For Indonesia, Pakistan, Bangladesh:
# 1. The population has slight crests and troughs with overall distributed data lying between 24-112 Million

# ### Plotting bargraphs to tally the gender-wise distribution among countries in the year 1960 and 2022:

# In[18]:


data_1960 = merged_data[merged_data["Year"] == 1960]               ## data of year 1960
data_2022 = merged_data[merged_data["Year"] == 2022]               ## data of year 2022

sns.set_style("darkgrid")
fig, axes = plt.subplots(1,2, figsize = (15,5))

graph_1960 = sns.barplot(x = data_1960["Country"], y = data_1960["Population"], hue = data_1960["Gender"], ax=axes[0], palette="rocket")
population1 = np.linspace(data_1960["Population"].min(), data_1960["Population"].max(), 13)
graph_1960.set_yticks(population1)
graph_1960.set_yticklabels(f"{pop/1000000:.2f}M" for pop in population1)
graph_1960.set_title("Gender-Wise Population distribution in 1960")


graph_2022 = sns.barplot(x = data_2022["Country"], y = data_2022["Population"], hue = data_2022["Gender"], ax=axes[1], palette="rocket")
population2 = np.linspace(data_2022["Population"].min(), data_2022["Population"].max(), 13)
graph_2022.set_yticks(population2)
graph_2022.set_yticklabels(f"{pop/1000000:.2f}M" for pop in population2)
graph_2022.set_title("Gender-Wise Population distribution in 2022")


plt.tight_layout()
plt.show()


# ### Oberservation From Above Charts:
# 
# - In 1960:
# 1. All the 5 above asian countries' population lied under 350 Million
# 2. India's female population was somewhat 314 Million which was much higher than male population of approx 220 Million
# 3. China's female population lied way below it's male population - female_pop: approx 207 Million | male_pop: approx 340 Million
# 4. Indonesia, Pakistan, Bangladesh had male-female ratio as somewhat equal with no major highs
# 
# - In 2022:
# 1. All the above 5 asian countries' population increased and lie under 740 Million
# 2. India's male population has seen an increase when compared to 1960, with a population of approx 730 Million
# 3. China's female population has seen an increase when compared to 1960, with a population of approx 680 Million
# 4. Indonesia, Pakistan and Bangladesh have seen an increase in population when compared to 1960, but the gender-wise distribution for each country lies somewhat equal with no major highs

# In[19]:


countries = merged_data["Country"].unique()

fig, axes = plt.subplots(3,2, figsize=(15,9))

sns.set_style("dark")
for i, country in enumerate(countries):
    data = merged_data[merged_data["Country"] == country]
    row = i // 2
    col = i % 2
    sns.lineplot(x = data["Year"], y = data["Population"], ax = axes[row, col], hue = data["Gender"])
    axes[row,col].set_title(country)
    years = np.linspace(data["Year"].min(), data["Year"].max(), 15)
    population = np.linspace(data["Population"].min(), data["Population"].max(), 6)

    ## setting the xticks and xtick labels
    axes[row, col].set_xticks(years)
    axes[row, col].set_xticklabels([str(int(year)) for year in years])

    ## setting the yticks and ytick labels
    axes[row,col].set_yticks(population)
    axes[row,col].set_yticklabels([f"{pop/1000000:.2f}M" for pop in population])

    # Adjust the x-axis limits based on the data range for the specific country
    axes[row, col].set_xlim(data["Year"].min(), data["Year"].max())

plt.grid(False)
plt.tight_layout()
plt.show()


# ### Oberservation From Above Charts:
# 
# - All the above countries have seen a increasing trend in their population number from 1960-2022
# - In India:
# 1. The female population was much more than the male population between the year 1960-2010
# 2. The male population surpassed the female population after the year 2013 and still going
# - In China:
# 1. From the year 1960 itself the chinese male population has always been high as compared to the female population
# - For Indonesia, Pakistan, Bangladesh:
# 1. The male and female population have both increased from the year 1960-2022 with an overall similar growth pattern

# In[20]:


countries = merged_data["Country"].unique()
year = 1960

fig, axes = plt.subplots(ncols=5, figsize = (15,15))

for i, country in enumerate(countries):
    data = merged_data[(merged_data["Country"] == country) & (merged_data["Year"] == year)]
    col = i
    axes[col].pie(data["Population"].values, labels = ["Male","Female"], autopct = "%1.1f%%", explode=[0.01,0.1], shadow=True)
    axes[col].set_title(country)

plt.tight_layout()

year = 2022

fig, axes = plt.subplots(ncols=5, figsize = (15,15))

for i, country in enumerate(countries):
    data = merged_data[(merged_data["Country"] == country) & (merged_data["Year"] == year)]
    col = i
    axes[col].pie(data["Population"].values, labels = ["Male","Female"], autopct = "%1.1f%%", explode=[0.01,0.1], shadow=True)
    axes[col].set_title(country)
    

plt.tight_layout()
plt.grid(False)
plt.show()


# ### ^The first pie chart is the male-female distribution in the year 1960
# ### The second pie chart is the male-female distribution in the year 2022

# ### Oberservation From Above Charts:
# 
# - Time Period : 1960 - 2022
# - In India:
# 1. The female population has decreased by 10% in the year 2022 as compared to the year 1960
# 2. The male population has increased by 10% in 2022 since the year 1960
# - In China:
# 1. The female population has increased 10.1% since the year 1960
# 2. The male population has seen a decrease by ~ 10% since the year 1960
# - In Pakistan:
# 1. The female population has increased by 3.6% in the year 2022 as compared to the year 1960
# 2. The male population has seen a decrease of 4.4% in the year 2022 as compared to the year 1960
# - For Indonesia and Bangladesh:
# 1. They have seen a somewhat equal male - female ratio in 2022 when compared with year 1960

# ## END NOTE: 
# 
# ### DATA SOURCE: https://data.worldbank.org/
# 
# ### In conclusion, the analysis of population data for various Asian countries from 1960 to 2022 reveals several significant insights:
# 
# ### Top 5 Most Populated Countries in 2022:
# 
# - India, China, Indonesia, Pakistan, and Bangladesh are the top five most populated Asian countries based on total population.
# - India and China also dominate the list based on male population, while China leads in terms of female population.
# 
# ### Population Distribution:
# 
# - India and China exhibit population distributions that surpass other countries, with variations between male and female populations.
# - India's male population is uniformly distributed, while China's male population shows a slight shift towards higher values.
# - In contrast, India's female population has a slight shift towards higher values, while China's female population is more evenly distributed.
# 
# ### Detailed Population Analysis:
# 
# - In India, the male population peaks between 200-300 million and the female population peaks between 500-650 million.
# - China's male population is right-skewed, primarily concentrated between 500-700 million, while the female population shows slight peaks around 200-280 million.
# - Indonesia, Pakistan, and Bangladesh have relatively uniform population distributions between 24-112 million.
# 
# ### Comparison of 1960 and 2022:
# 
# - In 1960, all five countries had populations under 350 million, with variations in male-female ratios.
# - By 2022, populations had increased, with India and China surpassing 700 million.
# - India's male population increased, while China's female population saw growth. 
# - Indonesia, Pakistan, and Bangladesh showed population increases with relatively balanced gender distributions.
# 
# ### Population Trends:
# 
# - All countries experienced population growth from 1960 to 2022.
# - India's female population exceeded the male population until 2010 when the trend reversed.
# - China consistently had a higher male population from 1960.
# - Indonesia, Pakistan, and Bangladesh displayed similar growth patterns with relatively balanced male-female ratios.
# 
# ### Population Changes from 1960 to 2022:
# 
# - In India, the female population decreased by 10% in 2022 compared to 1960, while the male population increased by 10%.
# - China saw a 10.1% increase in the female population but a 10% decrease in the male population.
# - Pakistan's female population increased by 3.6% in 2022, while the male population decreased by 4.4%.
# - Indonesia and Bangladesh showed somewhat equal male-female ratios in 2022 compared to 1960.
# 
# ### These insights provide valuable information for understanding population dynamics in Asian countries over the years. It is clear that India and China remain at the forefront of population statistics, with noteworthy shifts in gender distribution. These observations can inform further research and policy decisions related to population growth, gender demographics, and resource planning.
