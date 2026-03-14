import pandas as pd

# Load csv file
df = pd.read_csv("../../Data/CSV/censo_2023-01.xlsx - Datos.csv")

name = "Shihuahuaco"
species = df[df['NOMBRE_COMUN'].str.contains(name)]

print(species)
species.to_csv(f'SpeciesCSV/{name}.csv')