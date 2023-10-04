package micrograd

import (
	"fmt"
	"math/rand"
)

type Neuron struct {
	w      []*Value
	b      *Value
	nonlin bool
}

func NewNeuron(nin int, nonlin bool) *Neuron {
	neuron := &Neuron{w: make([]*Value, nin), b: NewValue(0.0), nonlin: nonlin}
	for i := range neuron.w {
		neuron.w[i] = NewValue((rand.Float64() * 2) - 1)
	}
	return neuron
}

func (n *Neuron) Forward(x []*Value) *Value {
	// check if the input is the same size as the weights
	if len(x) != len(n.w) {
		panic("input size mismatch")
	}
	sum := n.b
	for i := range x {
		sum = sum.Add(n.w[i].Mul(x[i]))
	}
	if n.nonlin {
		return sum.Relu()
	}
	return sum
}

func (n *Neuron) Parameters() []*Value {
	out := []*Value{n.b}
	for i := range n.w {
		out = append(out, n.w[i])
	}
	return out
}

func (n *Neuron) ZeroGrad() {
	for _, p := range n.Parameters() {
		p.Grad = 0.0
	}
}

func (n *Neuron) String() string {
	return fmt.Sprintf("Neuron(w=%v, b=%v)", n.w, n.b)
}
