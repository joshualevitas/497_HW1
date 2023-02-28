from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ffnnHeatmap = [[2311, 755], [643, 2405]]
sns.heatmap(ffnnHeatmap, annot=True, fmt='g')
plt.title("FFNN Confusion Matrix")
plt.show()