import tkinter as tk
from tkinter import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

root = tk.Tk()

# set window size
window_height = 900
window_width = 900

# set window position
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


bg = PhotoImage(file="logo.png")
my_label=Label(root,image=bg)
my_label.place(x=0,y=0 ,  relwidth=1, relheight=1)

teksti1 = StringVar()

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
    'figure.figsize': (18, 10),
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'}, font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)

covid = pd.read_csv('dataset.csv')

LARGE_FONT = ("Verdana", 12)


def func1():
    pd.set_option('display.max_columns',covid.shape[0]+1)
    print(covid)

def func2():
    for x in ["Agjensione lajmesh", "Agjensite e patundshmerise", "Agrikultura", "Aktivitete spitalore",
              "Aktivitete sportive", "Farmaceutike", "Furre e bukes", "Hotele dhe akomodimi", "Marketing",
              "Palestra e fitnesit", "Porodhimi me shumice", "Prodhimi i barnave",
              "Restaurante dhe aktivitete sherbyese",
              "Sherbime tjera ushqyese", "Sigurime", "Stomatologji", "Tekstil", "Transporti ajror",
              "Transporti urbane",
              "Tregti e pijeve"]:
        covid.UljeApoNgritje[covid.Sektori == x].plot(kind="kde")
    plt.title("Training Data - Ngritja apo Ulja ne baze te Sektorit")
    plt.legend(("Agjensione lajmesh", "Agjensite e patundshmerise", "Agrikultura", "Aktivitete spitalore",
                "Aktivitete sportive", "Farmaceutike", "Furre e bukes", "Hotele dhe akomodimi", "Marketing",
                "Palestra e fitnesit", "Porodhimi me shumice", "Prodhimi i barnave",
                "Restaurante dhe aktivitete sherbyese",
                "Sherbime tjera ushqyese", "Sigurime", "Stomatologji", "Tekstil", "Transporti ajror",
                "Transporti urbane",
                "Tregti e pijeve"))
    plt.show()

def func3():
    (covid.UljeApoNgritje.value_counts(normalize=True) * 100).plot.barh().set_title(
        "Training Data - Percentage of companies with positive GDP and negative")
    plt.show()

def func4():
    plt.scatter(covid['%assectorGDP(10 max)'], covid['Komuna'])
    plt.title('Sector GDP per Country')
    plt.xlabel('%assectorGDP(10 max)')
    plt.ylabel('Komuna')
    plt.xlim(-10, 10)
    plt.show()

def func5():
    le = LabelEncoder()
    covid_encoded = covid.iloc[:, 0:12]

    for i in covid_encoded:
        covid_encoded[i] = le.fit_transform(covid_encoded[i])
    X = covid_encoded.iloc[:, 0:11]
    y = covid_encoded.iloc[:, 4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    teksti = str(score)
    teksti1.set(teksti)

Label(root, text="Impact of Covid-19 in Economy of Kosovo", bg="white",font= "Helvetica 20" ).place(x = 160,y = 50)
Button(root, text="Print Dataset in terminal",bg="white",width='20' ,height='1',font="Raleway",command=func1).place(x = 100, y = 242)
Button(root, text="Increase and decrease based on sector",width='35' ,height='1',font="Raleway", bg="white",command=func2).place(x = 370, y = 242)
Button(root, text="Percentage of companies with positive and negative GDP",bg="white",width='50' ,height='1',font="Raleway",command=func3).place(x = 160, y = 300)
Button(root, text="Percentage of sector GDP per country",bg="white",width='50' ,height='1',font="Raleway",command=func4).place(x = 160, y = 350)
Button(root, text="Accuracy Score",bg="white",width='20' ,height='1',font="Raleway",command=func5).place(x = 300, y = 600)
Entry(root, textvariable=teksti1 , bg="white").place(width=600, height=80, x = 100, y = 450)
root.mainloop()
