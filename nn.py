from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
# irisデータセットを読み込む
iris = load_iris()
print(iris.data.shape)

# 分割する
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

model = MLPClassifier(solver='lbfgs', random_state=0, max_iter=3000)

# 学習する
model.fit(X_train,y_train)
# 予測する
pred = model.predict(X_test)
# モデルの保存
joblib.dump(model, 'nn.pkl', compress=True) # nn.pklというファイルが作成

print(model.score(X_test, y_test)) # 97%
print(y_test, pred)