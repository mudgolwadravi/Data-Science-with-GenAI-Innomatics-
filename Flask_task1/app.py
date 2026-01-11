from flask import Flask,request

app=Flask(__name__)


@app.route("/")
def home():
    return "Welcome to Home Page"

@app.route("/urlQueryParameter")
def url_query_parameter():
    if 'username' in request.args.keys():
        Username=request.args.get("username")

        return Username.upper()
    else:
        return "Query Parameter is missing!!!! \n Example : http://127.0.0.1:5000/urlQueryParameter?username=Surya"


if __name__=="__main__":
    app.run(debug=True)