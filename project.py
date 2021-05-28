import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from IPython.core.display import display, HTML
import seaborn as sns

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input style = "float:right" type="submit" value="Toggle code">''')

# Setting up visualisations
sns.set_style(style='white')
sns.set(rc={
    'figure.figsize': (12, 7),
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'}, font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)



covid = pd.read_csv('dataset.csv')
covid.info()
covid.describe()
print(covid)

df = pd.DataFrame(covid)
df.head()
print("--------------------------NULL VALUES------------------------")
df = df.fillna(df.mean())
print(df.isnull().sum())


print("--------------------------Sorting based on Population-------------------------------")
sort = df.sort_values(by='Popullsia2020', ascending=True)
print(sort.head())

print("--------------------------Sorting based on GDP-------------------------------")
sort = df.sort_values(by='%assectorGDP(10 max)', ascending=True)
print(sort.head())
(covid.UljeApoNgritje.value_counts(normalize=True) * 100).plot.barh().set_title\
    ("Training Data - Percentage of companies with positive GDP and negative")

plt.show()


fig_LlojiBiznesit = covid.LlojiIbiznesit.value_counts().plot.pie().legend(labels=["Biznes individual",
           "Shoqeri aksionare",
           "Shoqeri me pergjegjesi te kufizuara",
           "Dega e shoqerise se huaj",
           "Ortakeri e papergjithshme"],
           loc='center right', bbox_to_anchor=(2.9, 0.5)).set_title("Training Data -Llojet e Bizneseve")
plt.show()


for x in ["Agjensione lajmesh", "Agjensite e patundshmerise", "Agrikultura", "Aktivitete spitalore",
          "Aktivitete sportive", "Farmaceutike", "Furre e bukes", "Hotele dhe akomodimi", "Marketing",
          "Palestra e fitnesit", "Porodhimi me shumice", "Prodhimi i barnave", "Restaurante dhe aktivitete sherbyese",
          "Sherbime tjera ushqyese", "Sigurime", "Stomatologji", "Tekstil", "Transporti ajror", "Transporti urbane",
          "Tregti e pijeve"]:
    covid.UljeApoNgritje[covid.Sektori == x].plot(kind="kde")
plt.title("Training Data - Ngritja apo Ulja ne baze te Sektorit")
plt.legend(("Agjensione lajmesh", "Agjensite e patundshmerise", "Agrikultura", "Aktivitete spitalore",
            "Aktivitete sportive", "Farmaceutike", "Furre e bukes", "Hotele dhe akomodimi", "Marketing",
            "Palestra e fitnesit", "Porodhimi me shumice", "Prodhimi i barnave", "Restaurante dhe aktivitete sherbyese",
            "Sherbime tjera ushqyese", "Sigurime", "Stomatologji", "Tekstil", "Transporti ajror", "Transporti urbane",
            "Tregti e pijeve"))
plt.show()


plt.scatter(covid['%assectorGDP(10 max)'], covid['Komuna'])
plt.title('Sector GDP per Country')
plt.xlabel('%assectorGDP(10 max)')
plt.ylabel('Komuna')
plt.xlim(-10, 10)
plt.show()

plt.scatter(covid['%astotalGDP(10 max)'], covid['Komuna'])
plt.title('Total GDP per Country')
plt.xlabel('TotalGDP')
plt.ylabel('Komuna')
plt.xlim(0, 200)
plt.show()

plt.scatter(covid['%assectorGDP(10 max)'], covid['Sektori'])
plt.title('GDPperSector')
plt.xlabel('%assectorGDP(10 max)')
plt.ylabel('Sektori')
plt.xlim(-10, 10)
plt.show()

le = LabelEncoder()
covid_encoded = covid.iloc[:, 0:12]

for i in covid_encoded:
    covid_encoded[i] = le.fit_transform(covid_encoded[i])

print('------------------------ENCODED---------------------------------')
print(covid_encoded)

X = covid_encoded.iloc[:, 0:11]
y = covid_encoded.iloc[:, 4]

print('------------------------FROM HERE-------------------------------')
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('Accuracy : {:.2f}'.format(score))
