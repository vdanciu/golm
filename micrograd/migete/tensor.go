package migete

import "fmt"

type Tensor[T Number] struct {
	Data *TensorData[T]
	// Grad might need to be float
	Grad     *TensorData[T]
	prev     []any
	op       string
	backward func()
}

type ShapeType []int

func NewTensor[T Number](shape ShapeType, data []T, scalar bool) *Tensor[T] {
	td := NewTensorData(shape, data, scalar)
	return &Tensor[T]{Data: td, Grad: nil, prev: []any{}, op: "", backward: func() {}}
}

// Create a new empty tensor with a given shape and allocate memory for it
// based on the product of all dimensions given in the shape
func NewEmptyTensor[T Number](shape ShapeType) *Tensor[T] {
	td := NewTensorData(shape, make([]T, getSize(shape)), false)
	return &Tensor[T]{Data: td, Grad: nil, prev: []interface{}{}, op: "", backward: func() {}}
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

func (t *Tensor[T]) Reshape(ints ...int) *Tensor[T] {
	t.Data.Reshape(ints)
	return t
}

func (t *Tensor[T]) Index(by *Tensor[int]) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Index(by.Data), Grad: nil, prev: []interface{}{t, by}, op: "index", backward: func() {}}
}

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Add(other.Data), Grad: nil, prev: []interface{}{t, other}, op: "+", backward: func() {}}
}

func (t *Tensor[T]) Mul(other *Tensor[T]) *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Mul(other.Data), Grad: nil, prev: []interface{}{t, other}, op: "*", backward: func() {}}
}

func (t *Tensor[T]) Tanh() *Tensor[T] {
	return &Tensor[T]{Data: t.Data.Tanh(), Grad: nil, prev: []interface{}{t}, op: "tanh", backward: func() {}}
}

// func (t *Tensor[T]) CrossEntropy(other *Tensor[int]) *Tensor[T] {
// 	return &Tensor[T]{Data: t.Data.CrossEntropy(other.Data), Grad: nil, prev: []interface{}{t, other}, op: "cross_entropy", backward: func() {}}
// }

func (t *Tensor[T]) Softmax(dim int) *Tensor[float64] {
	return &Tensor[float64]{Data: t.Data.Softmax(dim), Grad: nil, prev: []interface{}{t}, op: "softmax", backward: func() {}}
}
