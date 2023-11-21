package main

import (
	"fmt"
	"vdanciu_lang_model/micrograd/migete"
)

func main() {
	t2 := migete.NewTensor(migete.FromData[float32]([][]float32{
		{1.0, 2.0, 3.0}, {2.0, 3.0, 5.0}, {3.0, 4.0, 3.0}}))
	t3 := t2.Neg()
	fmt.Printf("t3: %v\n", t3)
	t3.Backward()
	fmt.Printf("t2.grad %v\n", t2.Grad)
	fmt.Printf("t3.grad %v\n", t3.Grad)
}
