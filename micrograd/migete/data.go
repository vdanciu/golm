package migete

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
)

type Number interface {
	int | float64 | float32
}

type TensorData[T Number] struct {
	Data  []T
	Shape []int
	// Scalar is true if the tensor is a scalar
	Scalar bool
}

func Flatten[T Number](data any) []T {
	return flatten[T](reflect.ValueOf(data))
}

func MakeShape(dims ...int) []int {
	return dims
}

func NewTensorData[T Number](shape ShapeType, data []T, scalar bool) *TensorData[T] {
	return &TensorData[T]{Data: data, Shape: shape, Scalar: scalar}
}

func NewEmptyTensorData[T Number](shape ShapeType, scalar bool) *TensorData[T] {
	return NewTensorData[T](shape, make([]T, getSize(shape)), scalar)
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

func (l *TensorData[T]) Gather(dim int, indices *TensorData[int]) *TensorData[T] {
	if len(l.Shape) != len(indices.Shape) {
		panic("Gather: incompatible shapes (should have the same number of dimensions)")
	}

	for d, size := range l.Shape {
		if d != dim && size < indices.Shape[d] {
			panic("Gather: index should not be larger on any dimension other than the one we are gathering")
		}
	}
	/*
		out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
		out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
		out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
	*/
	out := make([]T, indices.Size())
	for i := 0; i < len(out); i++ {
		multiIdx := getMultiIndex(i, indices.Shape)
		fmt.Printf("multiIdx: %v\n", multiIdx)
		fmt.Printf("indices.Data: %v\n", indices.Data)
		multiIdx[dim] = indices.Data[i]
		dataIdx := getFlatIndex(multiIdx, l.Shape)
		fmt.Printf("shape: %v\n", l.Shape)
		fmt.Printf("dataIdx: %v\n", dataIdx)
		out[i] = l.Data[dataIdx]
	}
	return &TensorData[T]{Data: out, Shape: indices.Shape}
}

func (l *TensorData[T]) View(shape ShapeType) *TensorData[T] {
	if shape[0] == -1 {
		shape[0] = l.Size() / getSize(shape[1:])
	}
	if l.Size() != getSize(shape) {
		panic("View: incompatible shapes")
	}
	v := &TensorData[T]{Data: l.Data, Shape: shape}
	return v
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

func (p *TensorData[T]) Log() *TensorData[float64] {
	result := make([]float64, p.Size())
	for i := 0; i < p.Size(); i++ {
		result[i] = math.Log(float64(p.Data[i]))
	}
	return &TensorData[float64]{Data: result, Shape: p.Shape}
}

func (p *TensorData[T]) Neg() *TensorData[T] {
	result := make([]T, p.Size())
	for i := 0; i < p.Size(); i++ {
		result[i] = -p.Data[i]
	}
	return &TensorData[T]{Data: result, Shape: p.Shape}
}

func (p *TensorData[T]) Mean() *TensorData[float64] {
	size := float64(p.Size())
	var result float64
	for i := 0; i < p.Size(); i++ {
		result += float64(p.Data[i]) / size
	}
	return &TensorData[float64]{Data: []float64{result}, Shape: []int{1}}
}

func (p *TensorData[T]) CrossEntropy(target *TensorData[int]) *TensorData[float64] {
	softmax_probs := p.Softmax(1)
	target_view := target.View([]int{-1, 1})
	gather := softmax_probs.Gather(1, target_view)
	ilog := gather.Log().Neg()
	loss := ilog.Mean()
	return loss
}

func (p *TensorData[T]) Fill(value T) {
	for i := 0; i < p.Size(); i++ {
		p.Data[i] = value
	}
}

func flatten[T Number](data reflect.Value) []T {
	shape := make([]int, 0)
	val := data
	size := 1
	for val.Kind() == reflect.Slice {
		shape = append(shape, val.Len())
		size *= val.Len()
		val = val.Index(0)
	}
	if len(shape) == 1 {
		return data.Interface().([]T)
	} else if len(shape) > 1 {
		multiIdx := make([]int, len(shape)-1)
		result := make([]T, 0, size)
		for ; isValid(multiIdx, shape); next(multiIdx, shape) {
			val := data
			for i := 0; i < len(multiIdx); i++ {
				val = val.Index(multiIdx[i])
			}
			result = append(result, val.Interface().([]T)...)
		}
		return result
	}
	panic("flatten: invalid shape")
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
	if scalar {
		shape = []int{1}
		flatData = []T{data.(T)}
	} else if len(shape) == 1 {
		flatData = data.([]T)
	} else if len(shape) > 1 {
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
		if indices[i] >= shape[i] {
			panic("getFlatIndex: index out of range")
		}
		sp := 1
		for j := i + 1; j < len(indices); j++ {
			sp *= shape[j]
		}
		index += indices[i] * sp
	}
	fmt.Printf("getFlatIndex: index: %v, shape: %v, result: %v\n", indices, shape, index)
	return index
}

func getMultiIndex(index int, shape []int) []int {
	multiIndex := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		multiIndex[i] = index % shape[i]
		index /= shape[i]
	}
	return multiIndex
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

func isValid(indices, shape []int) bool {
	for i := 0; i < len(indices); i++ {
		if indices[i] <= shape[i]-1 {
			return true
		}
	}
	return false
}

func next(indices, shape []int) {
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < shape[i]-1 || i == 0 {
			indices[i]++
			return
		}
		indices[i] = 0
	}
}
