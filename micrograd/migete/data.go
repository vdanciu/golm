package migete

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"unsafe"
)

type Number interface {
	int | float64 | float32
}

type TensorBase interface {
	int | float64 | float32 |
		[]int | [][]int | [][][]int | [][][][]int |
		[]float64 | [][]float64 | [][][]float64 | [][][][]float64 |
		[]float32 | [][]float32 | [][][]float32 | [][][][]float32
}

func Flatten[T Number](data interface{}) []T {
	return flatten[T](reflect.ValueOf(data))
}

type MakeShapeFunc func(dims ...int) []int

func MakeShape(dims ...int) []int {
	return dims
}

type TensorData[T Number] struct {
	Data  []T
	Shape []int
	// Scalar is true if the tensor is a scalar
	Scalar bool
}

func NewTensorData[T Number](shape ShapeType, data []T, scalar bool) *TensorData[T] {
	return &TensorData[T]{Data: data, Shape: shape, Scalar: scalar}
}

func (t *TensorData[T]) String() string {
	return fmt.Sprintf("%v%v %v", t.Shape, reflect.TypeOf(t.Data), t.Data)
}

func (l *TensorData[T]) Size() int {
	return getSize(l.Shape)
}

func (l *TensorData[T]) Get(indices ...int) T {
	switch len(indices) {
	case 1:
		return l.Data[indices[0]]
	case 2:
		return l.Data[indices[0]*l.Shape[1]+indices[1]]
	case 3:
		return l.Data[indices[0]*l.Shape[1]*l.Shape[2]+indices[1]*l.Shape[2]+indices[2]]
	default:
		return l.Data[getFlatIndex(indices, l.Shape)]
	}
}

func (l *TensorData[T]) Reshape(shape ShapeType) *TensorData[T] {
	if l.Size() != getSize(shape) {
		panic("Reshape: incompatible shapes")
	}
	l.Shape = shape
	return l
}

func (l *TensorData[T]) Add(other *TensorData[T]) *TensorData[T] {
	if !reflect.DeepEqual(l.Shape, other.Shape) {
		if !l.Scalar && !other.Scalar {
			panic("Add: incompatible shapes")
		}
	}
	size := max(l.Size(), other.Size())
	result := make([]T, size)
	for i := 0; i < size; i++ {
		result[i] = l.Data[ternary[int](l.Scalar, 0, i)] + other.Data[ternary(other.Scalar, 0, i)]
	}

	return &TensorData[T]{Data: result, Shape: l.Shape}
}

/*
	From pytorch this are the rules of tensor multiplication:
	1.	If both tensors are 1-dimensional, the dot product (scalar) is returned.
	2.	If both arguments are 2-dimensional, the matrix-matrix product is returned.
	3.	If the first argument is 1-dimensional and the second argument is 2-dimensional,
	  	a 1 is prepended to its dimension for the purpose of the matrix multiply.
	  	After the matrix multiply, the prepended dimension is removed.
	4.	If the first argument is 2-dimensional and the second argument is 1-dimensional,
	  	the matrix-vector product is returned.
	5.	If both arguments are at least 1-dimensional and at least one argument is N-dimensional
	  	(where N > 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional,
	  	a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.
	  	If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the
	  	batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are
	  	broadcasted (and thus must be broadcastable). For example, if input is a (j×1×n×m)(j×1×n×m) tensor
	  	and other is a (k×m×p)(k×m×p) tensor, out will be an (j×k×n×p)(j×k×n×p) tensor.
	Broadcasting rules are:
	1.	If the number of dimensions of the operands does not match, they are prepended with 1s until
		until the number of dimensions match.
	2.	Then, the two tensors are said to be broadcastable if for each dimension pair they either match
		or one of them is 1. If the tensors are broadcastable, they are said to have compatible shapes.
	3.	After broadcasting, each tensor behaves as if it had shape equal to the elementwise maximum of
		shapes of the two input tensors.
	4.	In any dimension where one tensor had size 1 and the other tensor had size greater than 1,
		the first tensor behaves as if it were copied along that dimension
*/

func (t1 *TensorData[T]) Mul(t2 *TensorData[T]) *TensorData[T] {

	// check if both are 1-dimensional
	if len(t1.Shape) == 1 && len(t2.Shape) == 1 {
		return t1.mul1D(t2)
	}

	// check if both are 2-dimensional
	if len(t1.Shape) == 2 && len(t2.Shape) == 2 {
		if t1.Shape[1] != t2.Shape[0] {
			panic("Mul: incompatible shapes")
		}
		return t1.mul2D(t2)
	}

	// check if the first is 1-dimensional and the second is 2-dimensional
	if len(t1.Shape) == 1 && len(t2.Shape) == 2 {
		if t1.Shape[0] != t2.Shape[0] {
			panic("Mul: incompatible shapes")
		}
		return t1.mul1D2D(t2)
	}

	// check if the first is 2-dimensional and the second is 1-dimensional
	if len(t1.Shape) == 2 && len(t2.Shape) == 1 {
		if t1.Shape[1] != t2.Shape[0] {
			panic("Mul: incompatible shapes")
		}
		return t1.mul2D1D(t2)
	}

	// check if both are at least 1-dimensional and at least one is N-dimensional
	if len(t1.Shape) >= 1 && len(t2.Shape) >= 1 {
		if len(t1.Shape) > 2 && len(t2.Shape) > 2 {
			if t1.Shape[len(t1.Shape)-1] != t2.Shape[len(t2.Shape)-2] {
				panic("Mul: incompatible shapes")
			}
			return t1.mulND(t2)
		}
	}

	panic("Mul: not implemented for this case")
}

func (t1 *TensorData[T]) mul1D(t2 *TensorData[T]) *TensorData[T] {
	t1Scalar := false
	t2Scalar := false
	if t1.Shape[0] == 1 {
		t1Scalar = true
	} else if t2.Shape[0] == 1 {
		t2Scalar = true
	} else if t1.Shape[0] == t2.Shape[0] {

	} else {
		panic("Mul: incompatible shapes")
	}
	result := make([]T, max(t1.Size(), t2.Size()))
	for i := 0; i < len(result); i++ {
		result[i] = t1.Data[ternary[int](t1Scalar, 0, i)] *
			t2.Data[ternary[int](t2Scalar, 0, i)]
	}
	return &TensorData[T]{Data: result, Shape: []int{len(result)}}
}

func (t1 *TensorData[T]) mul2D(t2 *TensorData[T]) *TensorData[T] {
	shape := []int{t1.Shape[0], t2.Shape[1]}
	tr := make([]T, t1.Shape[0]*t2.Shape[1])
	// being flatten we can use the index to get the row and column
	for row := 0; row < t1.Shape[0]; row++ {
		for col := 0; col < t2.Shape[1]; col++ {
			for i := 0; i < t1.Shape[1]; i++ {
				tr[row*t2.Shape[1]+col] += t1.Data[row*t1.Shape[1]+i] * t2.Data[i*t2.Shape[1]+col]
			}
		}
	}
	return &TensorData[T]{Data: tr, Shape: shape}
}

func (t1 *TensorData[T]) mul1D2D(t2 *TensorData[T]) *TensorData[T] {
	/**
		If the first argument is 1-dimensional and the second argument is 2-dimensional,
	  	a 1 is prepended to its dimension for the purpose of the matrix multiply.
	  	After the matrix multiply, the prepended dimension is removed.
	*/
	t1m := *t1
	t1m.Shape = []int{1, t1.Shape[0]}
	result := t1m.mul2D(t2)
	result.Shape = result.Shape[1:]
	return result
}

func (t1 *TensorData[T]) mul2D1D(t2 *TensorData[T]) *TensorData[T] {
	/**
		If the first argument is 2-dimensional and the second argument is 1-dimensional,
	  	the matrix-vector product is returned.
	*/
	t2m := *t2
	t2m.Shape = []int{t2.Shape[0], 1}
	result := t1.mul2D(&t2m)
	result.Shape = result.Shape[1:]
	return result
}

func (t1 *TensorData[T]) mulND(t2 *TensorData[T]) *TensorData[T] {
	/**
		If the first argument is 2-dimensional and the second argument is 1-dimensional,
	  	the matrix-vector product is returned.
	*/
	panic("mulND: not implemented")
}

func (t *TensorData[T]) Index(by *TensorData[int]) *TensorData[T] {
	newShape := make([]int, 0)
	newShape = append(newShape, by.Shape...)
	if len(t.Shape) > 1 {
		newShape = append(newShape, t.Shape[1:]...)
	}
	result := make([]T, getSize(newShape))
	frameSize := getSize(t.Shape[1:])
	for i := 0; i < len(by.Data); i++ {
		for j := 0; j < frameSize; j++ {
			result[i*frameSize+j] = t.Data[by.Data[i]*frameSize+j]
		}
	}
	return &TensorData[T]{Data: result, Shape: newShape}
}

func (t *TensorData[T]) RandomizeNormal() {
	size := t.Size()
	for i := 0; i < size; i++ {
		t.Data[i] = (T)(rand.NormFloat64())
	}
}

func (t *TensorData[T]) RandomizeUniform(min, max int) {
	size := t.Size()
	for i := 0; i < size; i++ {
		t.Data[i] = (T)(rand.Intn(max-min) + min)
	}
}

func (t *TensorData[T]) Tanh() *TensorData[T] {
	result := make([]T, t.Size())
	for i := 0; i < t.Size(); i++ {
		result[i] = (T)(tanh[T](float64(t.Data[i])))
	}
	return &TensorData[T]{Data: result, Shape: t.Shape}
}

func (p *TensorData[T]) Softmax(dim int) *TensorData[float64] {
	if dim != len(p.Shape)-1 || len(p.Shape) > 2 {
		panic("Softmax: incompatible shapes")
	}
	result := make([]float64, p.Size())
	if len(p.Shape) == 2 {
		for row := 0; row < p.Shape[0]; row++ {
			softmax[T](p.Data, result, p.Shape[1], row*p.Shape[1])
		}
	} else {
		softmax[T](p.Data, result, p.Shape[0], 0)
	}

	return &TensorData[float64]{Data: result, Shape: p.Shape}
}

// func (p *TensorData[T]) CrossEntropy(y *TensorData[int]) *TensorData[float64] {
// 	if !reflect.DeepEqual(p.Shape[0:len(p.Shape)-1], y.Shape) {
// 		panic("CrossEntropy: incompatible shapes")
// 	}
// 	result := make([]float64, p.Shape[len(p.Shape)-1])
// 	for i := 0; i < len(result); i++ {
// 		p := p.Data[i*len(y.Data)+y.Data[i]]
// 		result[i] = -math.Log((float64)(p))
// 	}
// }

func flatten[T Number](data reflect.Value) []T {
	val := data
	size := 1
	for val.Kind() == reflect.Slice {
		size *= val.Len()
		val = val.Index(0)
	}
	var result []T
	if size == 1 && data.Kind() != reflect.Slice {
		result = []T{data.Interface().(T)}
	} else {
		fe := (*T)(unsafe.Pointer(val.Addr().Pointer()))
		result = unsafe.Slice(fe, size)
	}
	return result
}

func tanh[T Number](x float64) T {
	return (T)(math.Tanh(x))
}

func ternary[T any](cond bool, t, f T) T {
	if cond {
		return t
	}
	return f
}

func getSize(shape []int) int {
	s := 1
	for _, dim := range shape {
		s *= dim
	}
	return s
}

func FromData[T Number](data any) (shape ShapeType, flatData []T, scalar bool) {
	val := reflect.ValueOf(data)
	scalar = true
	for val.Kind() == reflect.Slice {
		scalar = false
		shape = append(shape, val.Len())
		val = val.Index(0)
	}
	if len(shape) == 1 {
		flatData = data.([]T)
	} else {
		flatData = flatten[T](reflect.ValueOf(data))
	}
	return shape, flatData, scalar
}

func getFlatIndex(indices []int, shape []int) int {
	if len(indices) != len(shape) {
		panic("getFlatIndex: incompatible shapes")
	}
	index := 0
	for i := 0; i < len(indices); i++ {
		sp := 1
		for j := i + 1; j < len(indices); j++ {
			sp *= shape[j]
		}
		index += indices[i] * sp
	}
	return index
}

func softmax[T Number](in []T, out []float64, size, offset int) {
	sum := 0.0
	for i := 0; i < size; i++ {
		out[offset+i] = math.Exp(float64(in[offset+i]))
		sum += out[offset+i]
	}
	for i := 0; i < size; i++ {
		out[offset+i] /= sum
	}
}
