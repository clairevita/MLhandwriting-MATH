import 'bootstrap/dist/css/bootstrap';
import * as tf from '@tensorflow/tfjs';

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
}