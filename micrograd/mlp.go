package micrograd

import "fmt"

type MLP struct {
	layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	depth := len(nouts) + 1
	sizes := make([]int, depth)
	sizes[0] = nin
	for i := range nouts {
		sizes[i+1] = nouts[i]
	}
	mlp := &MLP{make([]*Layer, len(nouts))}
	for i := range nouts {
		mlp.layers[i] = NewLayer(sizes[i], sizes[i+1], i != len(nouts)-1)
	}
	return mlp
}

func (l *MLP) Forward(x []*Value) []*Value {
	for _, layer := range l.layers {
		x = layer.Forward(x)
	}
	return x
}

func (l *MLP) Parameters() []*Value {
	parameters := make([]*Value, 0)
	for _, layer := range l.layers {
		parameters = append(parameters, layer.Parameters()...)
	}
	return parameters
}

func (l *MLP) ZeroGrad() {
	for _, p := range l.Parameters() {
		p.Grad = 0.0
	}
}

func (l *MLP) String() string {
	return fmt.Sprintf("Layers %v", l.layers)
}
