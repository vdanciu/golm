package migete

import (
	"math"
	"reflect"
	"runtime/debug"
	"testing"
)

// TODO: check the shapes of the result

func TestAdd(t *testing.T) {
	helperTestAdd[int](t, []int{10, 20}, 1, []int{11, 21})

	helperTestAdd[int](t, []int{10, 20}, []int{1, 2}, []int{11, 22})
	helperTestAdd[float64](t, []float64{10, 20}, []float64{1, 2}, []float64{11, 22})
	helperTestAdd[float32](t, []float32{10, 20}, []float32{1, 2}, []float32{11, 22})

	helperTestAdd[int](t, [][]int{{10, 20}, {10, 20}}, [][]int{{1, 2}, {1, 2}}, [][]int{{11, 22}, {11, 22}})
}

func TestMul(t *testing.T) {
	helperTestMul[int](t, []int{10, 20}, []int{1, 2}, []int{10, 40})
	helperTestMul[float64](t, []float64{10, 20}, []float64{1, 2}, []float64{10, 40})
	helperTestMul[float32](t, []float32{10, 20}, []float32{1, 2}, []float32{10, 40})

	helperTestMul[int](t, []int{10, 20}, []int{2}, []int{20, 40})
	helperTestMul[int](t, []int{10}, []int{1, 2}, []int{10, 20})
	helperTestMul[int](t, []int{10}, []int{2}, []int{20})

	helperTestMul[int](t,
		[][]int{{1, 2, 3}, {4, 5, 6}},
		[][]int{{10, 40}, {20, 50}, {30, 60}},
		[][]int{{140, 320}, {320, 770}})

	helperTestMul[int](t,
		[]int{1, 2, 3},
		[][]int{{10, 40}, {20, 50}, {30, 60}},
		[]int{140, 320})

	helperTestMul[int](t,
		[][]int{{1, 2, 3}, {4, 5, 6}},
		[]int{10, 20, 30},
		[]int{140, 320})
}

func TestIndex(t *testing.T) {
	t1 := NewTensor(FromData[int]([][]int{{1, 2, 3}, {4, 5, 6}}))
	t2 := NewTensor(FromData[int]([]int{0, 1, 0, 1, 0, 1}))
	ta := t1.Index(t2)
	correct := flatten[int](reflect.ValueOf([][]int{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}}))
	if !reflect.DeepEqual(ta.Data.Data, correct) {
		t.Errorf("Index failed, got %v, want %v", ta.Data.Data, correct)
	}
	if !reflect.DeepEqual(ta.Data.Shape, []int{6, 3}) {
		t.Errorf("Index shape failed, got %v, want %v", ta.Data.Shape, []int{6, 3})
	}
}

func TestSoftmax(t *testing.T) {
	t1 := NewTensor(FromData[float64]([][]float64{
		{10.0, 20.0, 30.0}, {20.0, 30.0, 50.0}, {30.0, 40.0, 30.0}}))
	t2 := t1.Softmax(1)
	ref := []float64{
		2.061060046209062e-09,
		4.539786860886666e-05,
		0.9999546000703311,
		9.357622949551801e-14,
		2.061153618190011e-09,
		0.9999999979387528,
		4.5395807829510914e-05,
		0.9999092083843408,
		4.5395807829510914e-05}
	if floatUnequal(t2.Data.Data, ref) {
		t.Errorf("Softmax failed, got %v, want %v", t2.Data.Data, ref)
	}
}

func helperTestAdd[T Number](t *testing.T, a, b, result any) {
	defer func() {
		if r := recover(); r != nil {
			// signal the error and the stack trace
			t.Errorf("Add failed with %v\n%v", r, string(debug.Stack()))
		}
	}()
	t1 := NewTensor(FromData[T](a))
	t2 := NewTensor(FromData[T](b))
	ta := t2.Add(t1)
	if !reflect.DeepEqual(ta.Data.Data, flatten[T](reflect.ValueOf(result))) {
		t.Errorf("Add failed, got %v, want %v", ta.Data.Data, result)
	}
}

func helperTestMul[T Number](t *testing.T, a, b, result any) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Add failed with %v\n%v", r, string(debug.Stack()))
		}
	}()
	t1 := NewTensor(FromData[T](a))
	t2 := NewTensor(FromData[T](b))
	ta := t1.Mul(t2)
	if !reflect.DeepEqual(ta.Data.Data, flatten[T](reflect.ValueOf(result))) {
		t.Errorf("Mul failed, got %v, want %v", ta.Data.Data, result)
	}
}

func floatUnequal(a, b []float64) bool {
	factor := 10000.0
	for i := range a {
		if math.Round(a[i]*factor)/factor != math.Round(b[i]*factor)/factor {
			return true
		}
	}
	return false
}
