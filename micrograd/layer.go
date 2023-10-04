package micrograd

import "fmt"

type Layer struct {
	neurons []*Neuron
}

func NewLayer(nin, nout int, nonlin bool) *Layer {
	layer := &Layer{neurons: make([]*Neuron, nout)}
	for i := range layer.neurons {
		layer.neurons[i] = NewNeuron(nin, nonlin)
	}
	return layer
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.neurons))
	for i := range out {
		out[i] = l.neurons[i].Forward(x)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	parameters := make([]*Value, 0)
	for _, neuron := range l.neurons {
		parameters = append(parameters, neuron.Parameters()...)
	}
	return parameters
}

func (l *Layer) ZeroGrad() {
	for _, p := range l.Parameters() {
		p.Grad = 0.0
	}
}

func (l *Layer) String() string {
	return fmt.Sprintf("Layer {in: %v, out: %v}", len(l.neurons[0].Parameters())-1, len(l.neurons))
}
