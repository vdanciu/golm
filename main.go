// package main

// import (
// 	"fmt"
// 	"vdanciu_lang_model/micrograd/migete"
// )

// func main() {
// 	//CreateDataset1("/Users/vdanciu/Downloads/lmd_matched")
// 	//RunMidi("midi/fixtures/new.mid")
// 	// runMoons()
// 	//RunNames()

// 	//testNewEngine()
// 	testMigete()
// }

// func testMigete() {
// 	t1 := migete.NewTensor([]int{10})
// 	t2 := migete.NewTensor([]int{1, 2})
// 	fmt.Printf("t1: %v\n", t1)
// 	fmt.Printf("t2: %v\n", t2)
// 	ta := t2.Mul(t1)
// 	fmt.Printf("ta: %v\n", ta)
// }

// // func testNewEngine() {
// // 	x := microtensor.NewEmptyTensor(microtensor.WithShape(0, 2, 2), microtensor.Float64)
// // 	y := microtensor.NewEmptyTensor(microtensor.WithShape(0), microtensor.Int)
// // 	x.Append([][]float64{{1, 2}, {3, 3}})
// // 	x.Append([][]float64{{4, 5}, {6, 6}})
// // 	y.Append(1)
// // 	y.Append(1)
// // 	y.Append(1)
// // 	y.Append(1)
// // 	fmt.Printf("x: %v\n", x)
// // 	fmt.Printf("y: %v\n", y)
// // }
