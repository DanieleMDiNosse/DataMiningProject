import seaborn as sns
plt.figure(figsize=(20,12))
plt.subplots_adjust(bottom=0.2)
heatmap = sns.heatmap(df.corr(), annot = True)
plt.savefig('MCHeatMap.png', dpi=1000)

mccat = pd.read_excel('MatriceCategorica.xlsx')
mccat.fillna(0, inplace=True)
mccat.index = mccat['Unnamed: 0']
mccat=mccat.drop(columns='Unnamed: 0')
mccat.rename_axis('Unnamed: 0').rename_axis('attributes', axis='columns')
mccat.rename_axis([None])

import matplotlib.colors
plt.figure(figsize=(20,12))
plt.subplots_adjust(bottom=0.2)

colors = sns.color_palette('rocket', 2)
levels = [0, 0.05]
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="max")
heatmap = sns.heatmap(mccat, annot = True, cmap = cmap, norm=norm)
heatmap.set_ylabel('')    
heatmap.set_xlabel('')
plt.savefig('CatHeatMap.png', dpi=1000)