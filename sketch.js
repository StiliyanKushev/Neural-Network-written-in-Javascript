let trainingData = [
    {
        inputs: [0,0],
        outputs: [0],
    },
    {
        inputs: [1,1],
        outputs: [0],
    },
    {
        inputs: [1,0],
        outputs: [1],
    },
    {
        inputs: [0,1],
        outputs: [1],
    },
];

let nn;

function setup(){
    createCanvas(800,800);
    nn = new NeuralNetwork(2, 1, 8, 1);
}

function draw(){
    background(200);
    noStroke();

    for(let i = 0; i < 1000; i++){
        let data = random(trainingData);
        nn.learn(data.inputs,data.outputs);
    }

    let res = 10;
    let cols = width / res;
    let rows = height / res;

    for(let i = 0; i < rows; i++){
        for(let j = 0; j < cols; j++){
            let x1 = i / cols;
            let x2 = j / rows;
            let inputs = [x1,x2];
            let y = nn.compute(inputs);
            fill(y * 255)
            rect(i * res, j * res, res, res);
        }
    }
}