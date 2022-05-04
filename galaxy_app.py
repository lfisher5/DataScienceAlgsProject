
from flask import Flask, jsonify, request
import pickle
from utils.myclassifiers import MyModifiedDecisionTreeClassifier
from utils.myrandomforestclassifier import MyRandomForestClassifier

app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>Welcome to The Galaxy Morphology Classifier Page</h1>'


@app.route("/predict", methods=["GET"])
def predict():
    # we need to parse the unseen instance's
    # attribute values from the request
    args = ['oiii_4959_flux', 'oiii_5007_flux', 'hei_5876_flux', 'oi_6300_flux',
            'h_alpha_flux', 'nii_6548_flux', 'sii_6717_flux', 'sii_6731_flux']
    inputs = []
    for arg in args:
        # empty string equals default value
        inputs.append(request.args.get(arg, ""))
    print(inputs)

    # TODO: fix the hardcoding
    prediction = predict_interviewed_well(inputs)
    # if anything goes wrong, predict_interviewed_well()
    # is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result)
    return "Error making prediction", 400


def predict_interviewed_well(instance):
    # we need to traverse the interview tree
    # and make a prediction for instan
    # how do we get the tree here?
    # generally, we need to save a trained
    # ML mode from another process for use later
    # (in this process)
    # enter pickling
    # unpickle tree.p
    infile = open("forest.p", "rb")
    header, forest = pickle.load(infile)
    infile.close()

    print("header:", header)

    # prediction time!!
    try:
        tree_obs = []
        for tree in forest:
            o = MyModifiedDecisionTreeClassifier()
            o.tree = tree
            print(tree)
            o.header = header
            tree_obs.append(o)
        print(tree_obs)
        forest_ob = MyRandomForestClassifier()
        forest_ob.forest = tree_obs
        instance = [float(val) for val in instance]
        print([instance])
        pred = forest_ob.predict([instance])
        print(pred)
        return pred

    except:
        print("error")
        return None


if __name__ == '__main__':
    app.run(debug=True)
