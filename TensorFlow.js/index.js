import 'bootstrap/dist/css/bootstrap';
import * as tf from '@tensorflow/tfjs';

import { MnistData } from './data';
import { deprecationWarn } from '@tensorflow/tfjs';

let model;

function createLogEntry(entry) {
    document.getElementById('log').innerHTML += '<br>' + entry;
}

function createModel() {
    createLogEntry('Creating Model ...');
    model = tf.sequential();
    createLogEntry('Model created.');

    createLogEntry('Adding layers...');
    //Input shape is the size of the image that is being passed into the neural network (px)
    //Kernel Size is the size of the shape that scans across the input (5x5 px)
    //Number of assessments windows that are applied to the input data through the kernel
    //Strides are the number of pixels that the filter windows jumps to for each input assessment (if it was 2 it would look at everother. 1 is the most secure)
    //Activation function being used is "Reactified Linear Unit". If there is content it outputs positive. If there is no content it outputs zero (in short).
    //Variance scaling enables the kernel to adapt to the shape that it's assessing.
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    //Pool size of the sliding window
    //Stides By how many pixels the pool slides over
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2],
    }))

    //We repeat the process for quality control with different parameters
    // We remove input shape because it is inherited from our 2d convolution layer
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2],
    }));

    //Flatten all of the output layers for further processing
    model.add(tf.layers.flatten());

    //Creates a dense fully connected layer for final assessment
    //Units are the number of characters we're searching for (0-9),
    //Adaptive shape for final assessment of full output
    //**Softmax for activation function uses probability for concluding what the most probable digit is being represented
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }));

    createLogEntry('Layers have been created.');

    createLogEntry('Starting compiling process...');

    //Next we're going to compile our model's losses and metrics. This is an assessment function to ensure total success. 
    //Optimizer used is a Stochastic  Gradient descent property, which eases out our compiling for QA
    //Calculates loss between final labels and the predictions
    model.compile({
        optimizer: tf.train.sgd(0.15),
        loss: 'categoricalCrossentropy'
    });

    createLogEntry('Compiling completed.');
}

let data;
// Next we're going to load the tensor flow MNIST data.
async function load() {
    createLogEntry('Loading MNIST data.');
    data = new MnistData();
    await data.load();
    createLogEntry('Data has been loaded.');
}

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

//Time to train the model
async function train() {
    createLogEntry('Initiating training.');
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        //Every time a training method happens, we're going to completely clear the batch
        const batch = tf.tidy(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;
        });

        await model.fit(
            batch.xs, batch.labels, { batchSize: BATCH_SIZE, epochs: 1 }
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }
    createLogEntry('Training completed.')
}

//This function addresses each of our training steps, creating the model, loading the data, and then training against it.
async function main(){
    createModel();
    await load();
    await train();
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Randomly Select Test Data and Predict";
}

//Now we can attempt a prediction versus our trained model
//
async function predict(batch){
    tf.tidy(() => {
        const input_value = Array.from(batch.labels.argMax(1).dataSync());

        const div = document.createElement('div');
        div.className = 'prediction-div';

        const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

        const prediction_value = Array.from(output.argMax(1).dataSync());
        const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const label = document.createElement('div');
        label.innerHTML = 'Original Value: ' + input_value
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;

        if (prediction_value - input_value == 0){
            label.innerHTML += '<br>Value recognized successfully.';
        } else {
            label.innerHTML += '<br>Recognition failed.';
        }

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}

//Non essential to project 3. This draws the image to the front end. Used for visualization as practice
function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j+ 0] = data[i] * 255;
        imageData.data[j+ 1] = data[i] * 255;
        imageData.data[j+ 2] = data[i] * 255;
        imageData.data[j+ 3] = data[i] * 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

document.getElementById('selectTestDataButton').addEventListener('click', async (el,ev) => {
    const batch = data.nextTestBatch(1);
    await predict(batch);
});

main();