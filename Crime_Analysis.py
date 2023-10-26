# import of library
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
import geopandas as gpd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.linear_model import LinearRegression
import time
from functools import wraps
import logging






# Configure logging
logging.basicConfig(
    filename='/Users/salahboughanmi/Desktop/Crime_Analysis/Crime_Analysis/DataVizPrj.log',  # Chemin complet ici
    filemode='a',  
    format='%(asctime)s - %(message)s',
    level=logging.INFO 
)
def log_execution_time(func):
    """
    Décorateur pour enregistrer le temps d'exécution et l'horodatage d'une fonction.
    """
    @wraps(func)  # Cela préserve les métadonnées de la fonction originale
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture du temps de début
        result = func(*args, **kwargs)  # Appel de la fonction originale
        end_time = time.time()  # Capture du temps de fin

        execution_time = end_time - start_time  # Calcul du temps d'exécution
        logging.info(f"Temps d'exécution: {execution_time:.2f} secondes")  # Enregistrement du temps d'exécution

        return result  # Retour du résultat de la fonction originale

    return wrapper

@log_execution_time
def expensive_function():
    time.sleep(2)

def display_time():
    st.divider()
    st.markdown("<h3 style='text-align: center;'>Execution Time </h3>", unsafe_allow_html=True)
    if st.button("Execute app"):
        expensive_function()





start_time = time.time()

@st.cache_data
# presentztion du projet
def presentation (): 
    st.markdown('## Data Viz Project')
    st.write('Communal and departmental bases for the main indicators of crimes recorded by the national police and gendarmerie (since 2016)')
    st.write("Link : [Government Data](https://www.data.gouv.fr/fr/datasets/bases-statistiques-communale-et-departementale-de-la-delinquance-enregistree-par-la-police-et-la-gendarmerie-nationales/)")   
    st.divider()

presentation()

@st.cache_data
#chargement et modif du CSV
def loadCSV ():
    # Read the  CSV file
    st.markdown("## Quick View of the dataframe")
    st.write('We convert the column annee and the column Code.département to string for better plotting')
    df = pd.read_csv("donnee-dep-data.gouv-2022-geographie2023-produit-le2023-07-17_cleaned.csv", sep=';')
    # Convert 'annee' and 'Code.département' to string for better plotting
    df['annee'] = df['annee'].astype(str)
    df['Code.département'] = df['Code.département'].astype(str)
    st.dataframe(df.head())
    return df

df_gouv = loadCSV()
st.divider()





@st.cache_data
def display_unique_crime_types(df_gouv):
    st.markdown('## All listed facts')
    # Let's check the unique values in the "classe" column to understand the different types of crimes
    unique_crime_types = df_gouv['classe'].unique()
    # Count the number of unique crime types
    num_unique_crime_types = len(unique_crime_types)
    # Utiliser Streamlit pour afficher les types uniques de crimes et leur nombre
    st.write("Unique types of crime : ", unique_crime_types)
    st.write("Number of unique crime types : ", num_unique_crime_types)

display_unique_crime_types(df_gouv)
st.divider()






@st.cache_data
# histogramme representant le nombre de crime par an
def plot_total_crimes_by_year(df_gouv):
    st.markdown('## Representative histogram of the number of crimes per year')
    # Regrouper les données par la colonne "annee" et sommer la colonne "faits" pour obtenir le nombre total de crimes pour chaque année
    total_crimes_by_year = df_gouv.groupby('annee')['faits'].sum().reset_index()
    # Utiliser Streamlit pour créer le graphique à barres
    st.bar_chart(total_crimes_by_year.set_index('annee'))

plot_total_crimes_by_year(df_gouv)
st.divider()







@st.cache_data
# histogramme negatif la variation du nombre de crime par an
def plot_altair_bar_chart(df_gouv):
    st.markdown('## Negative histogram representing the evolution of the number of crimes per year in %')

    # Suppression des valeurs NaN pour le graphique
    df_gouv.dropna(inplace=True)

    # Regroupement des données par 'annee' et somme des 'faits' pour chaque année
    total_crimes_by_year = df_gouv.groupby('annee')['faits'].sum().reset_index()

    # Calcul de l'évolution en pourcentage
    total_crimes_by_year['evolution_percentage'] = total_crimes_by_year['faits'].pct_change() * 100  # Calcul en %

    # Création du graphique en barres Altair pour l'évolution en %
    chart = alt.Chart(total_crimes_by_year).mark_bar().encode(
        x='annee:O',
        y='evolution_percentage:Q',
        color=alt.condition(
            alt.datum.evolution_percentage > 0,
            alt.value("green"),  # Le couleur si les valeurs sont positives
            alt.value("red")     # Le couleur si les valeurs sont négatives
        )
    ).properties(
        title="Percentage change in total number of crimes per year"
    )

    # Affichage du graphique dans Streamlit
    st.altair_chart(chart, use_container_width=True)

plot_altair_bar_chart(df_gouv)
st.divider()










# Debugged function
def plot_crime_evolution_debug(df_gouv):
    # Debug: Print or log the first few rows of df_gouv
    print("Initial DataFrame:")
    print(df_gouv.head())
    
    df_filtered = df_gouv[['annee', 'classe', 'faits']]
    
    df_grouped = df_filtered.groupby(['annee', 'classe'])['faits'].sum().reset_index()
    
    # Debug: Check if df_grouped is empty
    if df_grouped.empty:
        print("Error: df_grouped is empty")
        return
    
    # Debug: Print or log the first few rows of df_grouped
    print("Grouped DataFrame:")
    print(df_grouped.head())
    
    try:
        # Handle edge cases where x.iloc[0] could be zero or NaN
        df_grouped['percentage_change'] = df_grouped.groupby('classe')['faits'].apply(
            lambda x: (x / x.iloc[0]) * 100 - 100 if x.iloc[0] != 0 and not pd.isna(x.iloc[0]) else 0
        )
    except Exception as e:
        # Debug: Print or log the exception
        print(f"Error occurred: {e}")
        return
    
    # Debug: Print or log the first few rows of df_grouped after adding 'percentage_change'
    print("Grouped DataFrame with percentage_change:")
    print(df_grouped.head())
    















# observation du nobmre de crime par annee et departement
def plot_crimes_by_department(df_gouv):

    st.markdown('## Observation of the number of crimes by year and department ')

    df_total_agg = df_gouv.groupby('Code.département')['faits'].sum().reset_index()
    top_departments = df_total_agg.sort_values('faits', ascending=False).head(10)['Code.département'].tolist()

    unique_years = sorted(df_gouv['annee'].unique())
    unique_departments = sorted(df_gouv['Code.département'].unique())

    selected_departments = st.multiselect('Select the departments you want to see:', unique_departments, default=top_departments)
    selected_years = st.multiselect('Select the years you want to see:', unique_years)

    df_filtered = df_gouv[(df_gouv['annee'].isin(selected_years)) & (df_gouv['Code.département'].isin(selected_departments))]

    if len(selected_years) > 0:
        if len(selected_years) == 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            axs = [ax]
        else:
            if len(selected_years) > 2:
                nrows = 4
                ncols = 2
            else:
                nrows = len(selected_years)
                ncols = 1

            fig, axs = plt.subplots(nrows, ncols, figsize=(12 * ncols, 8 * nrows))
            axs = axs.flatten()

        for i, year in enumerate(selected_years):
            ax = axs[i]

            df_year = df_filtered[df_filtered['annee'] == year]
            df_year_agg = df_year.groupby('Code.département')['faits'].sum().reset_index()

            sns.barplot(x='Code.département', y='faits', data=df_year_agg.sort_values('faits', ascending=False), ax=ax)
            ax.set_title(f'Breakdown of incidents by department for the year {year}')
            ax.set_xlabel('Department code')
            ax.set_ylabel('Total number of incidents')

        if len(selected_years) > 1:
            for i in range(len(selected_years), nrows * ncols):
                axs[i].axis('off')

        st.pyplot(plt.gcf())
    else:
        st.write("Select at least one year to view graphs.")

plot_crimes_by_department(df_gouv)
st.divider()





@st.cache_data
#carte de la france 
def plot_crime_density(df_gouv, shapefile_path):
    st.markdown('## Crime density on the map of France ')
    # Group the crime dataset by department and sum the 'faits' (incidents)
    crime_by_department = df_gouv.groupby('Code.département').agg({'faits': 'sum'}).reset_index()
    crime_by_department['Code.département'] = crime_by_department['Code.département'].str.lstrip('0')

    # Load the France departments shapefile
    gdf_france_departments = gpd.read_file(shapefile_path)

    # Merge the geopandas dataframe with the crime_by_department dataframe
    merged = gdf_france_departments.set_index('code_insee').join(crime_by_department.set_index('Code.département'))

    # Filter out overseas departments
    metropolitan_departments = merged[~merged.index.str.startswith(('971', '972', '973', '974', '975', '976'))]

    # Plotting
    fig, ax = plt.subplots(1, figsize=(20, 20))

    y_min = metropolitan_departments.total_bounds[1] + 1  # Décalage vers le bas ici
    y_max = metropolitan_departments.total_bounds[3] 

    ax.set_xlim(metropolitan_departments.total_bounds[0] + 1, metropolitan_departments.total_bounds[2] - 1)
    ax.set_ylim(y_min, y_max)  # Limite de l'axe y modifiée ici

    metropolitan_departments.boundary.plot(ax=ax, linewidth=1, color='black')
    metropolitan_departments.plot(column='faits', ax=ax, legend=True, cmap='OrRd', linewidth=0.8, edgecolor='0.8', legend_kwds={'label': "Crime density by department"})

    for x, y, label in zip(metropolitan_departments.geometry.centroid.x, metropolitan_departments.geometry.centroid.y, metropolitan_departments.index):
        ax.text(x, y, label, fontsize=12, ha='center', va='center')

    plt.axis('off')

    st.pyplot(fig)
shapefile_path = "departements-20180101.shp"

plot_crime_density(df_gouv, shapefile_path)

st.divider()










def plot_crime_rate(df_gouv):
    st.markdown('## Crime rate per 1,000 inhabitants per year for departements ')
    # Convertir la colonne 'POP' en numérique
    df_gouv['POP'] = pd.to_numeric(df_gouv['POP'], errors='coerce')

    # Calculer le taux de crimes pour 1000 habitants pour chaque département et chaque année
    df_gouv['crime_rate_per_1000'] = (df_gouv['faits'] / df_gouv['POP']) * 1000

    # Agréger les données par année et par département
    agg_data = df_gouv.groupby(['annee', 'Code.département'])['crime_rate_per_1000'].sum().reset_index()

    # Trouver les départements avec le plus de crimes
    df_total_agg = df_gouv.groupby('Code.département')['faits'].sum().reset_index()
    top_departments = df_total_agg.sort_values('faits', ascending=False)['Code.département'].tolist()

    # Créer une checkbox pour sélectionner les départements
    selected_departments = st.multiselect('Select the departments you want to see:', top_departments, key='select_departments')

    # Filtrer les données agrégées pour inclure seulement les départements sélectionnés
    agg_data_selected_departments = agg_data[agg_data['Code.département'].isin(selected_departments)]

    # Créer le graphique pour les départements sélectionnés
    fig, ax = plt.subplots(figsize=(15, 8))
    colormap = plt.cm.get_cmap('tab10', len(selected_departments))

    for i, dept in enumerate(selected_departments):
        dept_data = agg_data_selected_departments[agg_data_selected_departments['Code.département'] == dept]
        ax.plot(dept_data['annee'], dept_data['crime_rate_per_1000'], label=f"Dept {dept}", color=colormap(i))

    plt.xlabel('Year')
    plt.ylabel('Crime rate per 1,000 inhabitants')
    plt.title('Crime rate per 1000 inhabitants per year for selected departments')
    plt.legend(loc='upper right', title='Départements')
    
    st.pyplot(fig)
plot_crime_rate(df_gouv)
st.divider()





# graph camembert 
def plot_pie_chart_streamlit(df):
    st.markdown('## Average distribution of crime types over all years')
    
    # Add radio button for user to select the data view
    choice = st.radio(
        'Select data view:',
        ('Average of all departments', 'Average of the 10 departments with the most events', 'Choose a department')
    )
    
    if choice == 'Average of the 10 departments with the most events':
        top_departments = df.groupby('Code.département')['faits'].sum().nlargest(10).index
        df = df[df['Code.département'].isin(top_departments)]
        
    elif choice == 'Choose a department':
        selected_department = st.selectbox('Choisir un département:', df['Code.département'].unique())
        df = df[df['Code.département'] == selected_department]

    grouped_by_classe_mean = df.groupby('classe')['faits'].mean().reset_index()
    grouped_by_classe_mean = grouped_by_classe_mean.sort_values('faits', ascending=False)

    plt.figure(figsize=(12, 12))
    plt.pie(grouped_by_classe_mean['faits'], labels=grouped_by_classe_mean['classe'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Average distribution of crime types over all years')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

plot_pie_chart_streamlit(df_gouv)
st.divider()







def plot_faceted_area_chart(df_cleaned):
    st.markdown('## Evolution of events over time by department')


    # Ajouter une case à cocher pour sélectionner le département
    selected_department = st.selectbox('Select a department:', df_cleaned['Code.département'].unique())
    
    # Filtrer les données pour inclure seulement le département sélectionné
    filtered_data = df_cleaned[df_cleaned['Code.département'] == selected_department]
    
    # Obtenir les types uniques de faits (crimes) pour ce département
    unique_crime_types = filtered_data['classe'].unique()
    
    # Ajouter des cases à cocher pour sélectionner les types de faits à analyser
    selected_crime_types = st.multiselect('Select the types of facts to be analyzed:', unique_crime_types, default=unique_crime_types)

    # Filtrer les données pour inclure seulement les types de faits sélectionnés
    filtered_data = filtered_data[filtered_data['classe'].isin(selected_crime_types)]

    # Pivoter les données pour s'assurer que nous avons toutes les combinaisons d'année et de type de fait
    pivoted_data = filtered_data.pivot(index='annee', columns='classe', values='faits').fillna(0)

    # Utiliser Streamlit pour créer le graphique en ligne
    st.line_chart(pivoted_data)

plot_faceted_area_chart(df_gouv)
st.divider()







# regression lineaire pour predire les crime en 23-24
def predict_incident_rates(df, department, feature='faits'):
    df_department = df[df['Code.département'] == department]
    df_department['annee'] = pd.to_numeric(df_department['annee'], errors='coerce')
    df_department[feature] = pd.to_numeric(df_department[feature], errors='coerce')
    df_department.dropna(subset=['annee', feature], inplace=True)
    X = df_department[['annee']]
    y = df_department[feature]
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array([[23], [24]])
    predictions = model.predict(future_years)
    adjusted_predictions = predictions + 1000
    return {'2023': adjusted_predictions[0], '2024': adjusted_predictions[1]}

def plot_and_predict_incident_rates(df, department, feature='faits'):
    df_department = df[df['Code.département'] == department]
    df_department['annee'] = pd.to_numeric(df_department['annee'], errors='coerce')
    df_department[feature] = pd.to_numeric(df_department[feature], errors='coerce')
    df_department.dropna(subset=['annee', feature], inplace=True)
    X = df_department[['annee']]
    y = df_department[feature]
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array([[23], [24]])
    predictions = model.predict(future_years)
    plt.figure(figsize=(10, 6))
    plt.bar(df_department['annee'], df_department[feature], color='blue', label='Historical Data')
    plt.bar(future_years.flatten(), predictions, color='orange', label='Predictions')
    plt.xlabel('Year')
    plt.ylabel(f'Number of {feature}')
    plt.title(f'{feature} by Year for Department {department}')
    plt.legend()
    plt.show()




# Streamlit app for Prediction and Plotting
st.title("Predict and Plot Incident Rates Per Departement")

selected_department = st.selectbox('Select Department', options=df_gouv['Code.département'].unique())

if st.button('Predict and Plot'):
    plot_and_predict_incident_rates(df_gouv, selected_department)
    st.pyplot()

    predictions = predict_incident_rates(df_gouv, selected_department)
    st.write(f"Predicted number of incidents for the year 2023: {predictions['2023']:.2f}")
    st.write(f"Predicted number of incidents for the year 2024: {predictions['2024']:.2f}")

    # Explanation of the output
    st.write("""
    ### Explanation of the Output
    
    The bar chart displays both the historical data and the predicted number of incidents for the years 2023 and 2024.
    The blue bars represent the actual number of incidents in previous years, while the orange bars are the predictions for 2023 and 2024.
    
    Please note that these are estimations based on a simple linear regression model and should not be considered as exact figures.
    """)





end_time = time.time()
execution_time = end_time - start_time
display_time()
st.write(f"Temps de chargement de l'application : {execution_time:.2f} secondes")


st.divider()

st.sidebar.header("Personal information :")
st.sidebar.write("BOUGHANMI")
st.sidebar.write("SALAH")
st.sidebar.write("2025")
st.sidebar.write("DE-1")
st.sidebar.write("GitHub : https://github.com/salahbgm")
st.sidebar.write("LinkedIn : www.linkedin.com/in/salah-boughanmi")
st.sidebar.write("#datavz2023efrei")