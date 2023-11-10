package migete

import "fmt"

type Tensor[T Number] struct {
	Data *TensorData[T]
	// Grad might need to be float
	Grad     *TensorData[T]
	prev     []Backwardable
	op       string
	backward func()
}

type ShapeType []int

type Backwardable interface {
	BackwardOne()
	GetPrev() []Backwardable
}

func NewTensor[T Number](shape ShapeType, data []T, scalar bool) *Tensor[T] {
	td := NewTensorData(shape, data, scalar)
	return &Tensor[T]{Data: td, Grad: nil, prev: []Backwardable{}, op: "", backward: func() {}}
}

// Create a new empty tensor with a given shape and allocate memory for it
// based on the product of all dimensions given in the shape
func NewEmptyTensor[T Number](shape ShapeType) *Tensor[T] {
	td := NewTensorData(shape, make([]T, getSize(shape)), false)
	return &Tensor[T]{Data: td, Grad: nil, prev: []Backwardable{}, op: "", backward: func() {}}
}

// Create a new tensor of the given shape with random numbers in a normal distribution
func NewRandomTensor[T Number](shape ShapeType) *Tensor[T] {
	randomTensor := NewEmptyTensor[T](shape)
	randomTensor.Data.RandomizeNormal()
	return randomTensor
}

func NewRandomUniformTensor[T Number](shape ShapeType, min, max int) *Tensor[T] {
	randomTensor := NewEmptyTensor[T](shape)
	randomTensor.Data.RandomizeUniform(min, max)
	return randomTensor
}

func (t *Tensor[T]) GetPrev() []Backwardable {
	return t.prev
}

// Appends a "frame" to the tensor where a frame's shape is
//
//	frameShape = tensorShape[1:]
//
// use Flatten to make sure the data is in the correct format
func (t *Tensor[T]) Append(data []T) {
	t.Data.Data = append(t.Data.Data, data...)
	t.Shape()[0]++
}

func (t *Tensor[T]) Size() int {
	return t.Data.Size()
}

func (t *Tensor[T]) Shape() ShapeType {
	return t.Data.Shape
}

func (t *Tensor[T]) String() string {
	return fmt.Sprintf("Tensor: %v (grad: %v)", t.Data, t.Grad)
}

func (t *Tensor[T]) ResetGrad() {
	t.Grad = nil
}

func (t *Tensor[T]) Reshape(ints ...int) *Tensor[T] {
	t.Data.Reshape(ints)
	return t
}

func (t *Tensor[T]) Index(by *Tensor[int]) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Index(by.Data), Grad: nil, prev: []Backwardable{t, by}, op: "index", backward: func() {}}
}

func (t *Tensor[T]) Gather(dim int, by *Tensor[int]) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Gather(dim, by.Data), Grad: nil, prev: []Backwardable{t, by}, op: "gather", backward: func() {}}
}

func (t *Tensor[T]) Get(indices ...int) T {
	return t.Data.Get(indices...)
}

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	out := &Tensor[T]{Data: t.Data.Add(other.Data), Grad: nil, prev: []Backwardable{t, other}, op: "+", backward: func() {}}
	out.backward = func() {
		initGradients[T](t, other, out)
		t.Grad = t.Grad.Add(out.Grad)
		other.Grad = other.Grad.Add(out.Grad)
	}
	return out
}

func (t *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	out := &Tensor[T]{Data: t.Data.MatMul(other.Data), Grad: nil, prev: []Backwardable{t, other}, op: "*", backward: func() {}}
	out.backward = func() {
		initGradients[T](t, other, out)
		t.Grad = t.Grad.Add(other.Data.MatMul(out.Grad))
		other.Grad = other.Grad.Add(t.Grad.MatMul(out.Grad))
	}
	return out
}

func (t *Tensor[T]) Tanh() *Tensor[T] {
	out := &Tensor[T]{Data: t.Data.Tanh(), Grad: nil, prev: []Backwardable{t}, op: "tanh", backward: func() {}}
	out.backward = func() {
		initGradients[T](t, out)
		t.Grad = t.Grad.Add(
			out.Grad.Mul(
				out.Data.Pow(2).
					Neg().
					Add(NewTensorData[T]([]int{}, []T{1}, true))))
	}
	return out
}

func (t *Tensor[T]) Softmax(dim int) *Tensor[float64] {
	return &Tensor[float64]{Data: t.Data.Softmax(dim), Grad: nil, prev: []Backwardable{t}, op: "softmax", backward: func() {}}
}

func (t *Tensor[T]) View(shape ...int) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.View(shape), Grad: nil, prev: []Backwardable{t}, op: "view", backward: func() {}}
}

func (t *Tensor[T]) Log() *Tensor[float64] {
	return &Tensor[float64]{Data: t.Data.Log(), Grad: nil, prev: []Backwardable{t}, op: "log", backward: func() {}}
}

func (t *Tensor[T]) Neg() *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Neg(), Grad: nil, prev: []Backwardable{t}, op: "neg", backward: func() {}}
}

func (t *Tensor[T]) Mean() *Tensor[float64] {
	return &Tensor[float64]{Data: t.Data.Mean(), Grad: nil, prev: []Backwardable{t}, op: "mean", backward: func() {}}
}

func (t *Tensor[T]) CrossEntropy(other *Tensor[int]) *Tensor[float64] {
	return &Tensor[float64]{Data: t.Data.CrossEntropy(other.Data), Grad: nil, prev: []Backwardable{t, other}, op: "cross_entropy", backward: func() {}}
}

func (t *Tensor[T]) BackwardOne() {
	t.backward()
}

func (t *Tensor[T]) Backward() {
	// gather all the tensors that need to be backpropagated
	tensors := []Backwardable{}
	visited := map[Backwardable]bool{}
	var buildPropList func(Backwardable)
	buildPropList = func(b Backwardable) {
		if _, ok := visited[b]; ok {
			return
		}
		visited[b] = true
		tensors = append(tensors, b)
		for _, t := range b.GetPrev() {
			buildPropList(t)
		}
	}
	buildPropList(t)
	initGradient[T](t)
	t.Grad.Fill(1)
	for _, tensor := range tensors {
		tensor.BackwardOne()
	}
}

func initGradient[T Number](t *Tensor[T]) {
	if t.Grad == nil {
		t.Grad = NewEmptyTensorData[T](t.Data.Shape, t.Data.Scalar)
	}
}

func initGradients[T Number](tensors ...*Tensor[T]) {
	for _, t := range tensors {
		initGradient[T](t)
	}
}
