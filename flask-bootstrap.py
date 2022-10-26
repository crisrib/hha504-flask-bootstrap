# Import package
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def bootstrap():
    return render_template('homeBs.html')

@app.route('/bootstrap2')
def template():
    return render_template('page2.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)