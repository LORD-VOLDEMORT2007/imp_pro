
//var var_model

$(function(){
    $('#test_button').prop('disabled', true);
    $('#train_model_button').prop('disabled', true);

    $("#predict_button_class1").click(function(){
        
        $('#predict_button_class1').prop('disabled', true);

        Collect_Button_pressed1  = {
            "buttonPressed1" : "true"
        }

        $.ajax({
            type: 'POST',
            url: "/images-collector1",
            data: JSON.stringify(Collect_Button_pressed1),
            dataType: "json",
            contentType: 'application/json',

            success: function (result) {
                
                images_captured1 = result.data
                
                if (images_captured1 === "True"){
                    $('#predict_button_class1').prop('disabled', false);
                    abled = 1
                    //post images from the folder class1
                    for (var i = 0; i < 100; i+=1){
                        $("#class1-images").prepend(
                            $('<img>',{id:'theImg',src:'./static/training_images/class1/frame' + i + '.jpg', width:30,height:30})
                            )
                    }
                }
            },
            error: function (result) {
                console.log(result.responseJSON.message)
            }
        });
    })
    
    $("#predict_button_class2").click(function(){
      
        $('#predict_button_class2').prop('disabled', true);


        Collect_Button_pressed2  = {
            "buttonPressed2" : "true"
        }
        //console.log("hello")
       
        $.ajax({
            type: 'POST',
            url: "/images-collector2",
            data: JSON.stringify(Collect_Button_pressed2),
            dataType: "json",
            contentType: 'application/json',

            success: function (result) {
                
                images_captured2 = result.data
                
                if (images_captured2 === "True"){
                    $('#predict_button_class2').prop('disabled', false);
                    if (abled === 1){
                        $('#train_model_button').prop('disabled', false);
                    }
                    //post images from the folder class1
                    for (var i = 0; i < 100; ++i){
                        $("#class2-images").prepend(
                            $('<img>',{id:'theImg2',src:"./static/training_images/class2/frame" + i + ".jpg", width:30,height:30})
                            )
                    }
                }
            },
            error: function (result) {
                console.log(result.responseJSON.message)
            }
        });


    })

    $("#train_model_button").click(function(){

        training_button = {
            "train_button_pressed" : "true"
        }

        $.ajax({
            type: 'POST',
            url: "/training",
            data: JSON.stringify(training_button),
            dataType: "json",
            contentType: 'application/json',

            success: function (result) {
                
                trained = result.data
                //var_model = result.model
                
                if (trained === "True"){
                    $('#test_button').prop('disabled', false);
                    
                }
            },
            error: function (result) {
                console.log(result.responseJSON.message)
            }
        });
    })

    $("#test_button").click(function(){

        test_button = {
            "test_button" : "true"
        }

        $.ajax({
            type: 'POST',
            url: "/testing",
            data: JSON.stringify(test_button),
            dataType: "json",
            contentType: 'application/json',

            success: function (result) {
                $("#prediction").html(result.data.predicted_output)
                $("#testing_img").attr('src', "./static/testing_images/frame0.jpg");

                
            },
            error: function (result) {
                console.log(result.responseJSON.message)
            }
        });

    })

})
