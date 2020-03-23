from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import joblib
import os


'''
predict関数・getName関数

predict関数
- nn.pyで学習して作成された学習済みモデルを読み込み、引数で受け取ったparametersを学習済みモデルに渡しています。
その結果[0]か[1]か[2]が変数predに代入されその値をreturnしています。

getName関数
- predict関数の返り値をgetName関数に渡すことで、ラベルから花の名前を返しています。'''


def predict(parameters):
    # モデル読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    print('params',params)
    print('params.shape',params.shape)
    pred = model.predict(params)
    print('pred',pred)
    return pred

# labelより花の名前取得
def getName(label):
    if label == 0:
        return 'Iris Setosa'
    if label == 1:
        return 'Iris Versicolor'
    if label == 2:
        return 'Iris Virginica'
    else:
        return 'Error'


app = Flask(__name__)
app.config.from_object(__name__)
secret_key = os.urandom(29)
app.config['SECRET_KEY'] = secret_key
# print(app.config['SECRET_KEY'])


# http://wtforms.simplecodes.com/docs/0.6/fields.html
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築

class IrisForm(Form):
    SepalLength = FloatField("Sepal Length(cm)（蕚の長さ）",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])

    SepalWidth  = FloatField("Sepal Width(cm)（蕚の幅）",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])

    PetalLength = FloatField("Petal length(cm)（花弁の長さ）",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])

    PetalWidth  = FloatField("petal Width(cm)（花弁の幅）",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form =IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        else: # 入力された値をそれぞれ変数に格納    
            SepalLength = float(request.form["SepalLength"])            
            SepalWidth  = float(request.form["SepalWidth"])            
            PetalLength = float(request.form["PetalLength"])            
            PetalWidth  = float(request.form["PetalWidth"])
            
            # それぞれの値を配列にする
            x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
            # 小さくてわかりづらいけどここでxを引数としてdef predictが実行されている
            pred = predict(x)
            # def getNameを実行している
            irisName = getName(pred)

            return render_template('result.html', irisName=irisName)
    elif request.method == 'GET':
        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()