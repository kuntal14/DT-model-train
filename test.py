import dask.dataframe as dd
import pandas as pd
import ast

# Create a sample pandas DataFrame
pdf = pd.DataFrame({'A': [[1.1,2],[3,4],[5,6]], 'B': [[1.1,2],[3,4],[5,6]]})

# Convert to Dask DataFrame with 2 partitions
ddf = dd.from_pandas(pdf, npartitions=1)
# print(ddf)

# Define a custom function to apply to each partition
# def custom_function(df):
#     print(1)
#     df['A'].apply(lambda x : ast.literal_eval(x))
#     return df
    
#     # return df.aaply(lambda x : ast.literal_eval(x))

  # Use map_partitions to apply the custom function to each partition
  result = ddf.map_partitions(custom_function)
  result = ddf['A'].apply(lambda x : ast.literal_eval(x))
print((ddf['A'][1].apply(lambda x : ast.literal_eval(x) ).compute()))

# # Compute the result
# computed_result = result.compute()
# print(computed_result)