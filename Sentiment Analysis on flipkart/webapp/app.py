from flask import Flask,request,render_template
import pickle
import re

app=Flask(__name__)

model=pickle.load(open('sentiment_model.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z\s]","",text)
    return text

@app.route('/',methods=['GET','POST'])
def index():
    prediction=None
    review_text=""

    if request.method=='POST':
        review_text=request.form['review']
        cleaned_text=clean_text(review_text)

        vector=vectorizer.transform([cleaned_text])
        result=model.predict(vector)[0]

        if result==0:
            prediction = "😡 Negative Review"
        else:
            prediction = "😊 Positive Review"
            
    return render_template('index.html',prediction=prediction,review=review_text)



if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)