// a complete reproduction of Andrej Karpathy's Micrograd in Go

package micrograd

import (
	"fmt"
	"math"
	"strings"
)

type Value struct {
	Data     float64
	Grad     float64
	prev     []*Value
	op       string
	backward func()
}

func NewValue(data float64) *Value {
	return &Value{Data: data, Grad: 0, prev: []*Value{}, op: "", backward: func() {}}
}

func makeValue(data float64, prev []*Value, op string) *Value {
	v := NewValue(data)
	v.prev = prev
	v.op = op
	v.backward = func() {}
	return v
}

func (v *Value) String() string {
	return fmt.Sprintf("[%f, grad: %f]", v.Data, v.Grad)
}

func (l *Value) Add(r *Value) *Value {
	out := makeValue(l.Data+r.Data, []*Value{l, r}, "+")
	out.backward = func() {
		l.Grad += out.Grad
		r.Grad += out.Grad
	}

	return out
}

func (l *Value) Mul(r *Value) *Value {
	out := makeValue(l.Data*r.Data, []*Value{l, r}, "*")
	out.backward = func() {
		l.Grad += r.Data * out.Grad
		r.Grad += l.Data * out.Grad
	}

	return out
}

func (l *Value) Pow(r float64) *Value {
	out := makeValue(math.Pow(l.Data, r), []*Value{l}, fmt.Sprintf("^%v", r))
	out.backward = func() {
		l.Grad += r * math.Pow(l.Data, r-1) * out.Grad
	}

	return out
}

func (l *Value) Relu() *Value {
	out := makeValue(math.Max(0, l.Data), []*Value{l}, "ReLU")
	out.backward = func() {
		if l.Data > 0 {
			l.Grad += out.Grad
		}
	}

	return out
}

func (l *Value) Neg() *Value {
	return l.Mul(NewValue(-1.0))
}

func (l *Value) Sub(r *Value) *Value {
	return l.Add(r.Neg())
}

func (l *Value) Div(r *Value) *Value {
	return l.Mul(r.Pow(-1.0))
}

func (l *Value) Backward() {
	// topological order all of the children in the graph
	topo := []*Value{}
	visited := map[*Value]bool{}
	var buildTopo func(*Value)
	buildTopo = func(v *Value) {
		if !visited[v] {
			visited[v] = true
			for _, child := range v.prev {
				buildTopo(child)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(l)

	l.Grad = 1.0
	// go in the reverse order of topo and call each backward
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

func (l *Value) PrintGraph() {
	fmt.Println(l)
	seen := map[*Value]bool{}
	var print func(*Value, int)
	print = func(v *Value, indent int) {
		if seen[v] {
			return
		}
		seen[v] = true
		leading := strings.Repeat(" ", indent)
		for _, child := range v.prev {
			fmt.Printf("%v %v -> %v\n", leading, v.op, child)
			print(child, indent+1)
		}
	}
	print(l, 0)
}
