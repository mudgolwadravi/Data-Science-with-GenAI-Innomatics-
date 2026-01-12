from flask import Flask,request,render_template
import re

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    matchs_strings=[]
    test_string="" 
    regx_pattern="" 

    if request.method=="POST":
        test_string=request.form.get("test_str")
        regx_pattern=request.form.get("regx_pattern")

        try:
            pattern=re.compile(regx_pattern)
            matchs_strings=pattern.findall(test_string)
        except re.error as e: 
            matchs_strings = [f"Regex error: {e}"]
    
    return render_template("index.html",matches=matchs_strings,test_string=test_string,regx_pattern=regx_pattern)


if __name__=="__main__":
    app.run(debug=True)