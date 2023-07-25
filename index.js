const { tensor2d } = require("@tensorflow/tfjs-node")
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs")
const imageData = fs.readFileSync('image.jpg');
// console.log("image data", imageData)
//opencv

// const imageRGBA = new Uint8Array(imageData.length);
// for (let i = 0; i < imageData.length; i++) {
//     imageRGBA[i] = imageData[i];
// }

// const imageTensor = tf.tensor(imageRGBA, [1, 224, 224,1]);
// console.log("tensor", imageTensor);




class AI {
    compile() {
        const model = tf.sequential()
        // input models
        model.add(tf.layers.dense({
            units: 3,
            inputShape: [3]
        }))

        // outpu model
        model.add(tf.layers.dense({
            units: 2
        }))

        model.compile({
            loss: "meanSquaredError",
            optimizer: "sgd"
        })

        return model

    }

    run() {

        const model = this.compile();

        const xy = tf.tensor2d([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ])
        const yx = tf.tensor2d([
            [1, 0],
            [0, 1],
            [1, 1]

        ])

        model.fit(xy, yx, {
            epochs: 800
        }).then(() => {
            const data = tf.tensor2d([
                [1.0, 1, 1.0]
            ])
            const prediction=model.predict(data)
            prediction.print();
        })
    }
}



const ai = new AI()
ai.run()


