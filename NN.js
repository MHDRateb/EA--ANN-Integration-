
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-|
//                                                                                   ||
// Created by:-                         MHD Rateb Alissa 2017                        ||  
//                           F21BC Biologically Inspired Computation                 ||  
//               Evolving static MLP using an EA to find the best set of weights     ||
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-|
// var synaptic = require('synaptic');


var population = [];
var child = [];
var network;

var populationSize;
var tournamentSize ;
var mutationRate ;
var EAiterations ;

// var mutationRate = 0.5;
//INPUT DATA
var Xor = [
    {
        input: [0, 0],
        output: 0
    },
    {
        input: [0, 1],
        output: 1
    },
    {
        input: [1, 0],
        output: 1
    },
    {
        input: [1, 1],
        output: 0
    }];
//__________________________________________________________________________________
//___________________________________Neural Network_______________________________________________

const NeuralNetwork = (initialNetwork) => {
    if (initialNetwork) {
        network = initialNeuralNetwork();
    }
    myNetwork = network.myNetwork;
    inputLayer = network.inputLayer;
    hiddenLayer = network.hiddenLayer;

    return createpopulation(myNetwork, inputLayer, hiddenLayer, population);
}

function initialNeuralNetwork() {

    var Neuron = synaptic.Neuron;
    var Layer = synaptic.Layer;
    var Network = synaptic.Network;
    var Trainer = synaptic.Trainer;

    inputLayer = new Layer(2);
    hiddenLayer = new Layer(5);
    outputLayer = new Layer(1);

    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    var myNetwork = new Network({
        input: inputLayer,
        hidden: [hiddenLayer],
        output: outputLayer
    });

    return { myNetwork: myNetwork, inputLayer: inputLayer, hiddenLayer: hiddenLayer }
}

function createpopulation(myNetwork, inputLayer, hiddenLayer, population) {

    if (population.length === 0) {
        //K HERE FOR POPULATIONS SIZE
        for (var k = 0; k < populationSize; k++) {
            //Create the zero generation (Intilize weigths first time randomly)
            var weightsSet = intilizeweigth(inputLayer, hiddenLayer);
            // these two lines return the neural and normal output for one individual
            //to pass these outputs to calculate the MSE(Mean sequre error)
            var outputs = [];
            outputs = calculateOutputs(myNetwork)
            //calculate mean sequre error and add it as attribute to the indvidual
            var meanerror = MSE(outputs);
            weightsSet.mean = meanerror;
            // add the indvidual to population
            population.push(weightsSet)
        }
    }
    //this block will excute after the first time(the initial time)
    //to apply the new weights and caluclte the means sequre error
    else {
        var weightsSet = applyWeigth(inputLayer, hiddenLayer, child);
        var outputs = [];
        outputs = calculateOutputs(myNetwork)
        var meanerror = MSE(outputs);
        weightsSet.mean = meanerror;
        // elitism function to compare the child with the population elements
        //and swap the child with the worst indivivdual in population if the child is better
        elitism(population, weightsSet);
    }
    return population;
}

function elitism(population, weightsSet) {
    var worst = getWorst(population);
    if (weightsSet.mean < worst.mean) {
        var index = population.indexOf(worst);
        population[index] = weightsSet;
    }
}
//this function return the worst indvivdual in the population
function getWorst(population) {
    var means = [];
    for (var i = 0; i < population.length; i++) {
        means.push(population[i].mean)
    }
    var max = Math.max.apply(null, means);
    var indx = means.indexOf(max);
    var worst = population[indx];
    return worst;
}
//intitla weights in first time
function intilizeweigth(inputLayer, hiddenLayer) {
    var weights = [];
    var objInput = inputLayer.connectedTo[0].connections;
    var objHidden = hiddenLayer.connectedTo[0].connections;

    Object.keys(objInput).forEach(function (key) {
        objInput[key].weight = getRandom();
        weights.push(objInput[key].weight);
    });

    Object.keys(objHidden).forEach(function (key) {
        objHidden[key].weight = getRandom();
        weights.push(objHidden[key].weight);
    });
    //return individual which will create the zero generation but without mean sequre error
    return { weights: weights }

}
//apply the child(which EA created it)
function applyWeigth(inputLayer, hiddenLayer, child) {
    var weights = [];
    var objInput = inputLayer.connectedTo[0].connections;
    var objHidden = hiddenLayer.connectedTo[0].connections;

    var i = 0;
    Object.keys(objInput).forEach(function (key) {
        objInput[key].weight = child[i];
        weights.push(objInput[key].weight);
        i += 1;
    });
    var h = 10;
    Object.keys(objHidden).forEach(function (key) {
        objHidden[key].weight = child[h];
        weights.push(objHidden[key].weight);
        h += 1;
    });
    return { weights: weights }

}
//return random number between -5 and 5
function getRandom() {
    var min = -5;
    var max = 5;
    var weight = Math.random() * (max - min) + min;
    return weight
}

//calculate outputs for each inputs with the weights from weightsSet
function calculateOutputs(myNetwork) {
    var outputs = [];
    for (var iii = 0; iii < Xor.length; iii++) {
        inputLayer.activate(Xor[iii].input);
        hiddenLayer.activate();

        var neuralOutput = outputLayer.activate();
        var actualoutput = Xor[iii].output;

        outputs.push({
            neuralOutput: neuralOutput[0],
            actualoutput: actualoutput
        })
    }

    return outputs
}
//Mean sequre error function
function MSE(outputs) {
    var mean = 0;
    outputs.forEach(function (output) {
        var difittestrence = Math.abs(output.actualoutput - output.neuralOutput);
        var error = Math.pow(difittestrence, 2);
        mean += error;
    })
    return mean / outputs.length;

}

//__________________________________________________________________________________
//_______________________________Genatic Algorithm___________________________________________________


var evolvePop = (population) => {
    for (var i = 0; i < EAiterations; i++) {
        //tournament selection to choose parent1 and 2
        var parent1 = parentSelection(population);
        var parent2 = parentSelection(population);
        //cross over from parent1 and parent2 to create child
        child = crossover(parent1, parent2);
        //mutate the child with take consideration mutationRate   
        mutate(child);
        //excute the neural network to calculate the fitness
        // for the child and then swap it
        //or discard it based on its fitness
        initialNetwork = false;
        NeuralNetwork(initialNetwork);
    }
}
//tournament selection to choose parent1 and 2
var parentSelection = (population) => {
    var subpopulation = [];
    for (var i = 0; i < tournamentSize; i++) {
        var randomIndex = Math.floor(Math.random() * population.length);
        var randomIndvid = population[randomIndex];
        subpopulation.push(randomIndvid);
    }
    var fittest = getFittest(subpopulation);
    return fittest;
}
//cross over function
var crossover = (parent1, parent2) => {
    var child = [];
    for (var i = 0; i < parent1.weights.length; i++) {
        child.push('$');
    }
    var startPos = Math.floor(Math.random() * parent1.weights.length);
    var endPos = Math.floor(Math.random() * parent1.weights.length);
    for (var i = 0; i < parent1.weights.length; i++) {
        if (startPos < endPos && i > startPos && i < endPos) {
            child.splice(i, 1, parent1.weights[i])
        }
        else if (startPos > endPos) {
            if (!(i < startPos && i > endPos)) {
                child.splice(i, 1, parent1.weights[i])
            }
        }
    }

    for (var i = 0; i < parent2.weights.length; i++) {
        if (!containWeigth(child, parent2.weights[i])) {
            for (var ii = 0; ii < parent2.weights.length; ii++) {
                if (child[ii] == "$") {
                    child.splice(ii, 1, parent2.weights[i])
                    break;
                }
            }
        }
    }

    return child;
}
//mutate function
var mutate = (weigthMean) => {
    var weigths = weigthMean;
    for (var i = 0; i < weigths.length; i++) {
        if (Math.random() < mutationRate) {

            var ii = Math.floor(weigths.length * Math.random());
            var temp = weigths[ii];
            weigths[ii] = weigths[i];
            weigths[i] = temp;
        }
    }
}
//fittest function to return the fittest indvidual from the population
function getFittest(population) {
    var means = [];
    for (var i = 0; i < population.length; i++) {
        means.push(population[i].mean)
    }
    var min = Math.min.apply(null, means);
    var indx = means.indexOf(min);
    var fittestt = population[indx];
    return fittestt;
}
var containWeigth = (child, obj) => {
    if (child.indexOf(obj) != -1) {
        return true
    }
    else {
        return false
    }
}

// main()
//__________________________________________________________________________________
//_______________________________Main___________________________________________________
function main() {
    var initialNetwork = true;
    NeuralNetwork(initialNetwork);

    console.log("individuals (weights sets) in zero generation ", population)
    //this function to test the best indivivdual in zero generation
    test(population);
    //excute the EA
    evolvePop(population);
    console.log("individuals (weights sets) in last generation ", population)
    //this function to test the best indivivdual in the last generation
    test(population);


}

//test function
function test(population) {

    var myNetwork = network.myNetwork;
    var inputLayer = network.inputLayer;
    var hiddenLayer = network.hiddenLayer;

    var fittest = getFittest(population);

    applyWeigth(inputLayer, hiddenLayer, fittest);

    function getFittest(population) {
        var means = [];
        for (var i = 0; i < population.length; i++) {
            means.push(population[i].mean)
        }
        var min = Math.min.apply(null, means);
        var indx = means.indexOf(min);
        var fittestt = population[indx];
        return fittestt;
    }
    console.log("best set of weights in this generation ", fittest)
    console.log("_______________");

    results.innerText =" execuation number  "+counter+"\n";
    results.innerText +="The fittest weights set in the last generation is "+"\n";   
    results.innerText +=JSON.stringify(fittest,null,2);
    result.appendChild(results);

    function applyWeigth(inputLayer, hiddenLayer, fittest) {

        var weights = [];
        var objInput = inputLayer.connectedTo[0].connections;
        var objHidden = hiddenLayer.connectedTo[0].connections;
        var i = 0
        Object.keys(inputLayer.connectedTo[0].connections).forEach(function (key) {
            inputLayer.connectedTo[0].connections[key].weight = fittest.weights[i];
            weights.push(inputLayer.connectedTo[0].connections[key].weight);
            i += 1;
        });
        var h = 10;
        Object.keys(hiddenLayer.connectedTo[0].connections).forEach(function (key) {
            hiddenLayer.connectedTo[0].connections[key].weight = fittest.weights[h];
            weights.push(hiddenLayer.connectedTo[0].connections[key].weight);
            h += 1;
        });

        return { weights: weights }

    }
    var Xor = [
        {
            input: [0, 0],
            output: 0
        },
        {
            input: [0, 1],
            output: 1
        },
        {
            input: [1, 0],
            output: 1
        },
        {
            input: [1, 1],
            output: 0
        }];
        results2.innerText ="input-output for execuation number  "+counter+"\n";
    for (var iii = 0; iii < Xor.length; iii++) {
        console.log(Xor[iii].input);
        results2.innerText +="  the input is "+Xor[iii].input+"\n";

        inputLayer.activate(Xor[iii].input);
        hiddenLayer.activate()
        var r = outputLayer.activate()
        console.log(r)
        results2.innerText +="  the neural output is "+r+ "\n";
        result2.appendChild(results2);
    }
}






