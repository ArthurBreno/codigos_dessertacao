import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

from scipy import stats

from matplotlib.colors import ListedColormap


# 1. Dados e Classes
classes = ['50%', '75%', '100%', '125%']

teste =[[	55,	13,	1	,11	],
 [	11	,55,	3	,8	],
 [	8	,11,	39,	20	],
 [	5	,6	,9	,62	]]

df = np.array(teste) # Matriz de Confusão como array NumPy

# 2. Definição da Função
def plot_confusion_matrix(cm, class_names, title='', cmap=plt.cm.Blues):
    #-------------------------------------------------------------------------
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    font_prop.set_size(18)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.size'] = 18
    #-------------------------------------------------------------------------
    
    
    plt.figure(figsize=(8, 6))
    img = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    cbar = plt.colorbar(img)
    
    
    for l in cbar.ax.get_yticklabels():
        l.set_fontproperties(fm.FontProperties(fname=font_path))
    
    
    
    # Rótulos (classes)
    tick_marks = np.arange(len(class_names))
    # Ajuste o 'ha' (horizontalalignment) para 'center' ou 'right' se 'rotation' for 0
    plt.xticks(tick_marks, class_names, rotation=0, ha='center', fontproperties=font_prop) 
    plt.yticks(tick_marks, class_names, fontproperties=font_prop)
    
    # Adicionar os valores dentro das células
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", fontproperties=font_prop ,color="white" if cm[i, j] > thresh else "black")
        
    plt.ylabel('Rotulo verdadeiro', fontproperties=font_prop)
    plt.xlabel('Rotulo predito', fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

# 3. Chamada da Função
plot_confusion_matrix(df, classes, title='')
