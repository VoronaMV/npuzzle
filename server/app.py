import json
from flask import Flask
from flask import render_template, jsonify


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html'), 200


@app.route('/get-data', methods=['post'])
def get_file_data():
    with open('result/res.json', 'r') as f:
        res = f.read()
    data = res
    data = json.loads(data)
    print(data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
