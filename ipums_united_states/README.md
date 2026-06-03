### Processing order:

1. Place the data file extracted from IPUMS (sample.dat) and the DDI (ddi.xml)
2. Create the exp_data, dados, models, and results folders, all with subfolders from 1 to 5 (except the dados folder, which stores the csv/parquet files)
3. Inside the main.py file: Run the pares() function to generate all the initial files
4. Process the necessary embeddings (separate notebook) and place them in the dados folder
5. Run the folds() function
6. Run the models() function to perform all the processing of the pairs, train, and run the model
