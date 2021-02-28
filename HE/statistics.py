import pandas as pd
df=  pd.read_csv('test.csv')
#df.groupby('image_name_list')

#for image,df_for_this_image in df.groupby('image_name_list'):
#print(image)
#print(df_for_this_image.shape)
time=0
for image,df_for_this_image in df.groupby('image_name_list'):
    time = time + 1
    df_for_this_image.sort_values('y_pred',inplace=True, ascending=False)
    if time < 3:	
        print(df_for_this_image.iloc[:10])
