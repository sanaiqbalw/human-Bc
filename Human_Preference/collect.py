from flask import Flask, render_template, flash, request, jsonify
from wtforms import Form, validators, StringField
import os, uuid
import json


def get_comparisons():
    json_file = os.path.join(SRC_DIR, SRC_FILE)
    with open(json_file, "r") as fp:
        comparisons = json.load(fp)
    return comparisons


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['JSON_AS_ASCII'] = False
SRC_DIR = "static/Collections"
SRC_FILE = "comparisons.json"
comps = get_comparisons()
labels = []
idx = 0


# app.run(host= '17.220.23.173', debug=False)

# @app.route('/autocomplete', methods=['GET'])
# def autocomplete():
#     search = request.args.get('q')
#     print(search)
#     results = DEV_TRIE.start_with_prefix(search.strip())
#
#     return jsonify(matching_results=results)
#
#
# @app.route('/fullsearch', methods=['GET'])
# def fullsearch():
#     search = request.args.get('q')
#     print(search)
#     results = df[df.Shanghainese.str.contains(search.strip())].Shanghainese.tolist()
#
#     return jsonify(matching_results=results)

@app.route('/startlabelling', methods=['GET'])
def serve_first_data():
    global idx
    print("Received first request")
    first_comp = comps[idx]
    idx += 1
    return jsonify(matching_results=first_comp, id=idx)


@app.route('/continuelabelling', methods=['GET'])
def serve_data():
    global idx
    # record choice
    past_choice = request.args.get('choice')
    comps[idx-1]['labels'] = past_choice
    print(past_choice)
    # serve next comparison

    print("Received subsequent requests")

    if idx==len(comps):
        # At the end of all the steps write labels to disk and show end
        json_file = os.path.join(SRC_DIR, SRC_FILE)
        with open(json_file, "w") as fp:
            json.dump(comps, fp, indent=4)
        return jsonify(matching_results=None, id=-1)

    comp = comps[idx]
    idx += 1
    return jsonify(matching_results=comp, id=idx)


@app.route('/')
def index():
    return render_template('display.html', comparisons=comps)


if __name__ == "__main__":
    port = os.environ.get('PORT', default=5000)
    if port:
        app.run(host='0.0.0.0', port=port)
    else:
        app.run()
