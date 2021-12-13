var sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
var reverseSigmoid = function (y) {
    y = Math.min(Math.max(y, 0.000000000000000000001), 0.9999999999999999999);
    return Math.log(y / (1 - y));
};
var sigmoidDerivative = function (x) { return sigmoid(x) * (1 - sigmoid(x)); };
var dSigmoid = function (x) { return x * (1 - x); };
var LR = 0.01;
var inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
];
var outputs = [
    0,
    1,
    1,
    0
];
var Link = /** @class */ (function () {
    function Link(source, target, strength) {
        this.source = source;
        this.target = target;
        this.strength = strength;
        this.thisTime = 0;
        this.target.inputs.push(this);
    }
    Link.prototype.apply = function () {
        var w = this.strength * this.source.output;
        this.target.input += w;
        this.thisTime = w;
    };
    return Link;
}());
var BiasNode = /** @class */ (function () {
    function BiasNode(output, target) {
        this.output = output;
        this.target = target;
    }
    BiasNode.prototype.apply = function () {
        for (var i = 0; i < this.target.length; i++) {
            var link = this.target[i];
            link.apply();
        }
    };
    return BiasNode;
}());
var Neuron = /** @class */ (function () {
    function Neuron(target, bias) {
        this.target = target;
        this.input = 0;
        this.output = 0;
        this.biasNode = null;
        this.inputs = [];
        this.propagated = false;
        this.error = 0;
        if (bias) {
            this.biasNode = new BiasNode(1, []);
            this.biasNode.target = [new Link(this.biasNode, this, Math.random())];
        }
    }
    Neuron.prototype.clear = function () {
        this.input = 0;
        this.output = 0;
    };
    Neuron.prototype.apply = function (value) {
        this.propagated = false;
        this.error = 0;
        if (this.biasNode !== null) {
            this.biasNode.apply();
        }
        if (value == undefined) {
            this.output = sigmoid(this.input);
        }
        else {
            this.output = this.input;
        }
        for (var i = 0; i < this.target.length; i++) {
            var link = this.target[i];
            link.apply();
        }
    };
    return Neuron;
}());
var NeuralNetwork = /** @class */ (function () {
    function NeuralNetwork(input, hidden) {
        this.end = new Neuron([], false);
        this.output = new Neuron([], false);
        this.output.target = [new Link(this.output, this.end, 1)];
        this.hidden = [];
        for (var i = 0; i < hidden.length; i++) {
            this.hidden.unshift([]);
            for (var j = 0; j < hidden[i]; j++) {
                var neuron = new Neuron([], j == 0);
                this.hidden[0].push(neuron);
                if (i === 0) {
                    neuron.target.push(new Link(neuron, this.output, Math.random()));
                }
                else {
                    for (var k = 0; k < hidden[i - 1]; k++) {
                        neuron.target.push(new Link(neuron, this.hidden[1][k], Math.random()));
                    }
                }
            }
        }
        this.input = [];
        for (var i = 0; i < input; i++) {
            var neuron = new Neuron([], false);
            this.input.push(neuron);
            for (var j = 0; j < this.hidden[0].length; j++) {
                neuron.target.push(new Link(neuron, this.hidden[0][j], Math.random()));
            }
        }
    }
    NeuralNetwork.prototype.run = function (input) {
        for (var i = 0; i < this.input.length; i++) {
            this.input[i].input = input[i];
            this.input[i].apply(false);
        }
        for (var i = 0; i < this.hidden.length; i++) {
            for (var j = 0; j < this.hidden[i].length; j++) {
                this.hidden[i][j].apply();
            }
        }
        this.output.apply();
        return this.end.input;
    };
    NeuralNetwork.prototype.backpropagate = function (node, targetOutput) {
        // targetOutput = reverseSigmoid(targetOutput);
        var actualOutput = node.output;
        if (node.propagated) {
            return;
        }
        node.propagated = true;
        if (targetOutput !== actualOutput) {
            var error = void 0;
            if (targetOutput === undefined) {
                error = node.error;
            }
            else {
                error = actualOutput - targetOutput;
            }
            node.error = error;
            var linkSum = node.inputs.reduce(function (value, link) { return value + link.strength; }, 0);
            for (var i = 0; i < node.inputs.length; i++) {
                var link = node.inputs[i];
                if (link.source instanceof Neuron) {
                    link.source.error = (link.strength / linkSum) * error;
                }
                link.strength += error * dSigmoid(link.thisTime) * LR;
            }
        }
    };
    NeuralNetwork.prototype.train = function (input, targetOutput) {
        for (var i = 0; i < this.input.length; i++) {
            this.input[i].input = input[i];
            this.input[i].apply(false);
        }
        for (var i = 0; i < this.hidden.length; i++) {
            for (var j = 0; j < this.hidden[i].length; j++) {
                this.hidden[i][j].apply();
            }
        }
        this.output.apply();
        var actualOutput = this.end.input;
        console.log('error: ' + (targetOutput - actualOutput));
        this.backpropagate(this.output, targetOutput);
        // if (targetOutput !== actualOutput) {
        // let error = targetOutput - actualOutput;
        // let sum = this.hidden[this.hidden.length - 1].reduce((sum, neuron) => sum + neuron.target[0].strength, 0);
        // for (let i = 0; i < this.hidden[this.hidden.length - 1].length; i++) {
        // this.hidden[this.hidden.length - 1][i].target[0].strength += error * sum * 0.05;
        // }
        // }
    };
    NeuralNetwork.prototype.plot = function (ctx) {
        function circle(x, y, r) {
            ctx.moveTo(x + r, y);
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
        }
        function line(x1, y1, x2, y2, size) {
            var f = ctx.fillStyle;
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
        for (var i = 0; i < this.hidden.length; i++) {
            for (var j = 0; j < this.hidden[i].length; j++) {
                ctx.fillStyle = '#88f';
                circle(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), 10);
                ctx.fillStyle = '#000';
                ctx.fillText(this.hidden[i][j].output.toFixed(2) + '', ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30));
                if (i == this.hidden.length - 1) {
                    line(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), ctx.canvas.width / 2, 10, this.hidden[i][j].target[0].strength * 10);
                }
                else {
                    for (var k = 0; k < this.hidden[i][j].target.length; k++) {
                        line(ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - (i * 30), ctx.canvas.width / 2 + (30 * k) - (this.hidden[(i)].length * 15), 50 + this.hidden.length * 30 - ((i + 1) * 30), this.hidden[i][j].target[k].strength * 10);
                    }
                }
            }
        }
        for (var i = 0; i < this.input.length; i++) {
            ctx.fillStyle = '#88f';
            circle(ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30), 10);
            ctx.fillStyle = '#000';
            ctx.fillText(this.input[i].output.toFixed(2) + '', ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30));
            for (var j = 0; j < this.input[i].target.length; j++) {
                line(ctx.canvas.width / 2 + (30 * i) - (this.input.length * 15), 80 + (this.hidden.length * 30), ctx.canvas.width / 2 + (30 * j) - (this.hidden[i].length * 15), 50 + this.hidden.length * 30 - ((0) * 30), this.input[i].target[j].strength * 10);
            }
        }
    };
    return NeuralNetwork;
}());
var net = new NeuralNetwork(2, [3, 3, 3, 3]);
console.log(net.run([1, 1]));
var canvas = document.querySelector('#canvas');
var ctx = canvas.getContext('2d');
net.plot(ctx);
function train() {
    for (var i = 0; i < inputs.length; i++) {
        net.train(inputs[i], outputs[i]);
    }
    net.run([1, 0]);
    ctx.clearRect(0, 0, 500, 500);
    net.plot(ctx);
}
setInterval(train, 100);
