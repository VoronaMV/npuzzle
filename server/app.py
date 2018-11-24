import json
from itertools import chain
from flask import Flask
from flask import render_template, jsonify


app = Flask(__name__)


@app.route('/')
def hello_world():
    with open('result/res.json', 'r') as f:
        res = f.read()
    data = res
    data = json.loads(data)
    initial_state = list()
    _len = None
    for row in data['moves'][0]:
        _len = len(row)
        initial_state.extend(row)
    return render_template('index.html', state=initial_state, len=_len), 200


@app.route('/get-data', methods=['post'])
def get_file_data():
    with open('result/res.json', 'r') as f:
        res = f.read()
    data = res
    data = json.loads(data)
    for i, elem in enumerate(data['moves']):
        data['moves'][i] = list(chain(*elem))
    print(data)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
