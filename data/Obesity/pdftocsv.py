import tabula

# Read the table from PDF and convert to CSV
tabula.convert_into("datamine/data/Obesity/prevalence-of-adult-overweight-obesity-2EN.pdf", "Obesity.csv", output_format="csv", pages='all')
