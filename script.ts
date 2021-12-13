const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const reverseSigmoid = (y: number) => {
    y = Math.min(Math.max(y, 0.000000000000000000001), 0.9999999999999999999);
    return Math.log(y / (1 - y));
}

const sigmoidDerivative = (x: number) => sigmoid(x) * (1 - sigmoid(x));
const dSigmoid = (x: number) => x * (1 - x);

const LR = 0.01;

const inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
];
const outputs = [
    0,
    1,
    1,
    0
];

class Link {
    thisTime = 0;
    constructor(public source: Neuron | BiasNode, public target: Neuron, public strength: number) {
        this.target.inputs.push(this);
    }
    apply() {
        const w = this.strength * this.source.output;
        this.target.input += w;
        this.thisTime = w;
    }
}

class BiasNode {
    constructor(public output: number, public target: Link[]) { }

    apply() {
        for (let i = 0; i < this.target.length; i++) {
            const link = this.target[i];
            link.apply();
        }
    }
}

class Neuron {
    input = 0;
    output = 0;
    biasNode?: BiasNode = null;
    inputs: Link[] = [];
    propagated = false;
    error = 0;
    constructor(public target: Link[], bias: boolean) {
        if (bias) {
            this.biasNode = new BiasNode(1, []);
            this.biasNode.target = [new Link(this.biasNode, this, Math.random())];
        }
    }
    clear() {
        this.input = 0;
        this.output = 0;
    }
    apply(value?: boolean) {
        this.propagated = false;
        this.error = 0;
        if (this.biasNode !== null) {
            this.biasNode.apply();
        }
        if (value == undefined) {
            this.output = sigmoid(this.input);

        } else {
            this.output = this.input;
        }
        for (let i = 0; i < this.target.length; i++) {
            const link = this.target[i];
            link.apply();
        }
    }
}

class NeuralNetwork {
    input: Neuron[];
    hidden: Neuron[][];
    output: Neuron;
    end: Neuron;

    constructor(input: number, hidden: number[]) {
        this.end = new Neuron([], false);
        this.output = new Neuron([], false);
        this.output.target = [new Link(this.output, this.end, 1)];
        this.hidden = [];
        for (let i = 0; i < hidden.length; i++) {
            this.hidden.unshift([]);
            for (let j = 0; j < hidden[i]; j++) {
                let neuron = new Neuron([], j == 0);
                this.hidden[0].push(neuron);
                if (i === 0) {
                    neuron.target.push(new Link(neuron, this.output, Math.random()));
                } else {
                    for (let k = 0; k < hidden[i - 1]; k++) {
                        neuron.target.push(new Link(neuron, this.hidden[1][k], Math.random()));
                    }
                }
            }
        }
        this.input = [];
        for (let i = 0; i < input; i++) {
            let neuron = new Neuron([], false);
            this.input.push(neuron);
            for (let j = 0; j < this.hidden[0].length; j++) {
                neuron.target.push(new Link(neuron, this.hidden[0][j], Math.random()));
            }
        }
    }

    run(input: number[]) {
        for (let i = 0; i < this.input.length; i++) {
            this.input[i].input = input[i];
            this.input[i].apply(false);
        }
        for (let i = 0; i < this.hidden.length; i++) {
            for (let j = 0; j < this.hidden[i].length; j++) {
                this.hidden[i][j].apply();
            }
        }
        this.output.apply();
        return this.end.input;
    }

    backpropagate(node: Neuron, targetOutput?: number) {
        // targetOutput = reverseSigmoid(targetOutput);
        let actualOutput = node.output;
        if (node.propagated) { return; }
        node.propagated = true;
        if (targetOutput !== actualOutput) {
            let error;
            if (targetOutput === undefined) {
                error = node.error;
            } else {
                error = actualOutput - targetOutput;
            }
            node.error = error;
            const linkSum = node.inputs.reduce((value, link) => value + link.strength, 0);
            for (let i = 0; i < node.inputs.length; i++) {
                const link = node.inputs[i];
                if (link.source instanceof Neuron) {
                    link.source.error = (link.strength / linkSum) * error;
                }
                link.strength += error * dSigmoid(link.thisTime) * LR;
            }
        }
    }

    train(input: number[], targetOutput: number) {
        for (let i = 0; i < this.input.length; i++) {
            this.input[i].input = input[i];
            this.input[i].apply(false);
        }
        for (let i = 0; i < this.hidden.length; i++) {
            for (let j = 0; j < this.hidden[i].length; j++) {
                this.hidden[i][j].apply();
            }
        }
        this.output.apply();
        let actualOutput = this.end.input;
        console.log('error: ' + (targetOutput - actualOutput));
        this.backpropagate(this.output, targetOutput);
        // if (targetOutput !== actualOutput) {
        // let error = targetOutput - actualOutput;
        // let sum = this.hidden[this.hidden.length - 1].reduce((sum, neuron) => sum + neuron.target[0].strength, 0);
        // for (let i = 0; i < this.hidden[this.hidden.length - 1].length; i++) {
        // this.hidden[this.hidden.length - 1][i].target[0].strength += error * sum * 0.05;
        // }
        // }
    }

    plot(ctx: CanvasRenderingContext2D) {
        function circle(x, y, r) {
            ctx.moveTo(x + r, y);
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
        }
        function line(x1: number, y1: number, x2: number, y2: number, size: number) {
            let f = ctx.fillStyle;
            ctx.lineWidth = size;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            ctx.fillStyle = f;
        }
        ctx.fillStyle = '#88f';
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.4)';
        circle(ctx.canvas.width / 2, 10, 10);
        ctx.fillStyle = '#000';
        ctx.fillText(this.end.input.toFixed(2) + '', ctx.canvas.width / 2, 10);
        for (let i = 0; i < this.hidden.length; i++) {
            for (let j = 0; j < this.hidden[i].length; j++) {
                ctx.fillStyle = '#88f';
                circle(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), 10);
                ctx.fillStyle = '#000';
                ctx.fillText(this.hidden[i][j].output.toFixed(2) + '', ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30));
                if (i == this.hidden.length - 1) {
                    line(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), ctx.canvas.width / 2, 10, this.hidden[i][j].target[0].strength * 10);
                } else {
                    for (let k = 0; k < this.hidden[i][j].target.length; k++) {
                        line(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), ctx.canvas.width / 2 + (30 * k) - (this.hidden[(i)].length * 15), 50 + this.hidden.length * 30 - ((i + 1) * 30), this.hidden[i][j].target[k].strength * 10);
                    }
                }
            }
        }

        for (let i = 0; i < this.input.length; i++) {
            ctx.fillStyle = '#88f';
            circle(ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30), 10);
            ctx.fillStyle = '#000';
            ctx.fillText(this.input[i].output.toFixed(2) + '', ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30));
            for (let j = 0; j < this.input[i].target.length; j++) {
                line(ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30), ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - ((0) * 30), this.input[i].target[j].strength * 10);
            }
        }
    }
}

let net = new NeuralNetwork(2, [3, 3, 3, 3]);
console.log(net.run([1, 1]));

const canvas = <HTMLCanvasElement>document.querySelector('#canvas');
const ctx = canvas.getContext('2d');

net.plot(ctx);

function train() {
    for (let i = 0; i < inputs.length; i++) {
        net.train(inputs[i], outputs[i]);
    }
    net.run([1, 0]);
    ctx.clearRect(0, 0, 500, 500);
    net.plot(ctx);
}

setInterval(train, 100);