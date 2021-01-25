import 'bootstrap/dist/css/bootstrap';
import * as tf from '@tensorflow/tfjs';

import { MnistData } from './data';
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
        filter: 8,
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
        filter: 16,
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
        loss: 'cetgoricalCrossentropy'
    });

    createLogEntry('Compiling completed.');


}