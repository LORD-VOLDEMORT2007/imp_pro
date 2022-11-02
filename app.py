from flask import Flask , render_template , request , jsonify
from image_ai import *

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/images-collector1" , methods = ["POST"])
def class1_captured_images():
    Collect_Button_pressed1 = request.json.get("buttonPressed1")
    print(Collect_Button_pressed1)
    if Collect_Button_pressed1 != "true":
        return jsonify({
            "status":"error",
            "message":"upload some images yo"
        })
    else:
        captured1 = collector1(Collect_Button_pressed1)
        return jsonify({
            "status" : "success",
            "data" : captured1
        })

@app.route("/images-collector2" , methods = ["POST"])
def class2_captured_images():
    Collect_Button_pressed2 = request.json.get("buttonPressed2")
    print(Collect_Button_pressed2)
    if Collect_Button_pressed2 != "true":
        return jsonify({
            "status":"error",
            "message":"upload some images yo"
        })
    else:
        captured2 = collector2(Collect_Button_pressed2)
        return jsonify({
            "status" : "success",
            "data" : captured2
        })

@app.route("/training", methods = ["POST"])
def training_images():
    Train_button_pressed = request.json.get("train_button_pressed")
    print(Train_button_pressed)
    if Train_button_pressed != "true":
        return jsonify({
            "status" : "error",
            "message" : "click the damn train button"
        })
    else:
        Model_trained = train(Train_button_pressed)
        print(Model_trained)
        
        return jsonify({
            "status" : "success",
            "data" : Model_trained,
            
        })

@app.route("/testing", methods = ["POST"])
def testing_data():
    Test_button_pressed = request.json.get("test_button")
    print(Test_button_pressed)
    if Test_button_pressed != "true":
        return jsonify({
            "status" : "error",
            "message" : "click the damn train button"
        })
    else:
        predicted_output = testing()
        print(predicted_output)
        return jsonify({
            "status" : "success",
            "data" : predicted_output,
            
        })

    



if __name__  ==  "__main__":
    app.run(debug = True)

#in the testing function call the train function in that model pass that 
#as the param to the test function