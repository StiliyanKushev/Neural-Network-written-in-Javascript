class NeuralNetwork {
    constructor(countInputs, countHiddenLayers, countHiddenNodes, countOutputs){
        // init the counts of layers and nodes
        this.countInputs = countInputs;
        this.countHiddenLayers = countHiddenLayers;
        this.countHiddenNodes = countHiddenNodes;
        this.countOutputs = countOutputs;

        // other
        this.activationFunction = this.sigmoid();
        this.learningRate = 0.1;
        
        // create the matrices
        this.inputs = new Matrix(countInputs, 1);
        this.outputs = new Matrix(countOutputs, 1);
        this.hidden = new Array(countHiddenLayers).fill(new Matrix(countHiddenNodes, 1));

        //genereate weights
        this.inputWeights = new Matrix(countHiddenNodes, countInputs);
        this.hiddenWeights = new Array(countHiddenLayers - 1).fill(new Matrix(countHiddenNodes));
        this.outputWeights = new Matrix(countOutputs, countHiddenNodes);
        this.inputWeights.randomize();
        this.outputWeights.randomize();
        this.hiddenWeights.map(e => e.randomize());
    }

    // generate a random deciamal between "min" and "max"
    static rand(min, max){
        return Math.random() * (max - min) + min;
    }

    //#region activation functions
    sigmoid(){
        let f = (x) => 1 / (1 + Math.pow(Math.E,-x));
        let d = (x) => f(x) * (1 - f(x));
        return new ActivationFunction(f,d);
    }
    //#endregion

    feedforward(){
        // input -> hidden[0]
        this.hidden[0] = Matrix.multiply(this.inputWeights, this.inputs);
        this.hidden[0].map(this.activationFunction.func);
        
        // hidden[i - 1] -> hidden[i]
        for(let i = 1; i < this.countHiddenLayers; i++){
            this.hidden[i] = Matrix.multiply(this.hiddenWeights[i - 1], this.hidden[i - 1]);
            this.hidden[i].map(this.activationFunction.func);
        }

        // hidden[length - 1] -> outputs
        this.outputs = Matrix.multiply(this.outputWeights, this.hidden[this.countHiddenLayers - 1]);
        this.outputs.map(this.activationFunction.func);
    }

    learn(inputs, answers){
        if(answers.length != this.countOutputs) throw new Error("Answers must be the same length as outputs.");
        // calculate the output layer
        this.compute(inputs);

        // calculate the output errors
        let outputErrors = Matrix.fromArrayV(answers).subtract(this.outputs);

        // calculate the hidden errors
        let hiddenErrors = new Array(this.countHiddenLayers).fill([]);
        hiddenErrors[this.countHiddenLayers - 1] = Matrix.multiply(this.outputWeights.transpose(), outputErrors);
        for(let i = this.countHiddenLayers - 2; i >= 0; i--){
            hiddenErrors[i] = Matrix.multiply(this.hiddenWeights[i].transpose(), hiddenErrors[i + 1]);
        }

        // create the error function based on the activation function
        let errFunc = (e) =>  this.learningRate * e * this.activationFunction.dfunc(e);

        // calculate output deltas
        let outputDeltas = Matrix.multiply(Matrix.map(outputErrors.multiply(this.outputs),errFunc), this.hidden[this.countHiddenLayers - 1].transpose());
        // update output weights
        this.outputWeights = this.outputWeights.add(outputDeltas);

        // TODO hidden layers multiple

        // calculate input deltas
        let inputDeltas = Matrix.multiply(Matrix.map(hiddenErrors[0].multiply(this.hidden[0]),errFunc), this.inputs.transpose());
        // update input weights
        this.inputWeights = this.inputWeights.add(inputDeltas);
    }

    compute(inputs){
        if(inputs.length != this.countInputs) throw new Error("invalid number of inputs");

        this.inputs = Matrix.fromArrayV(inputs);
        this.feedforward();
        return this.outputs.toArray();
    }
}

class ActivationFunction {
    constructor(func, dfunc){
        this.func = func;
        this.dfunc = dfunc;
    }
}

class Matrix {
    constructor(rows,cols){
        this.rows = rows;
        this.cols = cols || rows;
        this.matrix = new Array(rows);
        for(let i = 0; i < rows; i++){
            this.matrix[i] = new Array(cols);
            for(let j = 0; j < cols;j++){
                this.matrix[i][j] = 0;
            }
        }
    }

    static fromArrayH(arr){
        let m = new Matrix(1, arr.length);
        m.matrix = [arr];
        return m;
    }
    static fromArrayV(arr){
        let m = new Matrix(arr.length, 1);
        for(let i = 0; i < arr.length; i++) m.matrix[i][0] = arr[i];
        return m;
    }

    transpose(){
        let newMatrix = new Matrix(this.cols, this.rows);
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < this.cols;j++){
                newMatrix.matrix[j][i] = this.matrix[i][j];
            }
        }
        return newMatrix;
    }

    subtract(m2){
        if(m2.rows != this.rows || m2.cols != this.cols) throw new Error("Cannot subtract the matrix.");
        let newMatrix = new Matrix(m2.rows, m2.cols);
        for(let i = 0; i < m2.rows; i++){
            for(let j = 0; j < m2.cols;j++){
                newMatrix.matrix[i][j] = this.matrix[i][j] - m2.matrix[i][j];
            }
        }
        return newMatrix;
    }

    add(m2){
        if(m2.rows != this.rows || m2.cols != this.cols) throw new Error("Cannot add the matrix.");
        let newMatrix = new Matrix(m2.rows, m2.cols);
        for(let i = 0; i < m2.rows; i++){
            for(let j = 0; j < m2.cols;j++){
                newMatrix.matrix[i][j] = this.matrix[i][j] + m2.matrix[i][j];
            }
        }
        return newMatrix;
    }

    toArray(){
        if(this.rows == 1) return this.matrix[0];
        else if(this.cols == 1){
            let arr = [];
            for(let i = 0; i < this.rows;i++) arr[i] = this.matrix[i][0];
            return arr;
        }
        else throw new Error("Cannot convert matrix to array");
    }

    print(){
        console.table(this.matrix);
    }

    randomize(){
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < this.cols;j++){
                this.matrix[i][j] = NeuralNetwork.rand(-1,1);
            }
        }
    }

    map(func){
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < this.cols;j++){
                this.matrix[i][j] = func(this.matrix[i][j]);
            }
        }
        return this;
    }

    static map(m,func){
        for(let i = 0; i < m.rows; i++){
            for(let j = 0; j < m.cols;j++){
                m.matrix[i][j] = func(m.matrix[i][j]);
            }
        }
        return m;
    }

    multiply(m2){
        let newMatrix = new Matrix(this.rows, m2.cols);
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < m2.cols;j++){
                newMatrix.matrix[i][j] = this.matrix[i][j] * m2.matrix[i][j];
            }
        }
        return newMatrix;
    }

    static multiply(m1, m2){
        let newMatrix = new Matrix(m1.rows, m2.cols);
        for(let i = 0; i < m1.rows; i++){
            for(let j = 0; j < m2.cols;j++){
                for(let k = 0; k < m2.rows; k++){
                    newMatrix.matrix[i][j] += m1.matrix[i][k] * m2.matrix[k][j];
                }
            }
        }
        return newMatrix;
    }
}